"""Warp cover artwork onto the three blank books in reference/source2.png,
preserving the photo's original lighting/shadow via a multiply-blend shading map."""
import sys
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

SOURCE = "reference/source2.png"

# top-left, top-right, bottom-right, bottom-left corners of each book's front cover face
BOOKS = {
    "left":   [(88, 436), (404, 433), (405, 917), (91, 920)],
    "center": [(473, 433), (787, 432), (789, 920), (472, 921)],
    "right":  [(851, 432), (1176, 432), (1174, 921), (850, 920)],
}


def warp_into_quad(cover_img, quad, canvas_size):
    """Perspective-warp cover_img to fill quad, return RGBA image of canvas_size."""
    w, h = cover_img.size
    src_quad = [(0, 0), (w, 0), (w, h), (0, h)]
    coeffs = find_coeffs(src_quad, quad)
    warped = cover_img.convert("RGBA").transform(
        canvas_size, Image.PERSPECTIVE, coeffs, Image.BICUBIC
    )
    return warped


def find_coeffs(target_pts, source_pts):
    matrix = []
    for p1, p2 in zip(target_pts, source_pts):
        matrix.append([p2[0], p2[1], 1, 0, 0, 0, -p1[0] * p2[0], -p1[0] * p2[1]])
        matrix.append([0, 0, 0, p2[0], p2[1], 1, -p1[1] * p2[0], -p1[1] * p2[1]])
    A = np.array(matrix, dtype=np.float64)
    B = np.array(target_pts, dtype=np.float64).reshape(8)
    res = np.linalg.solve(A, B)
    return res.tolist()


def polygon_mask(quad, size):
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).polygon(quad, fill=255)
    return mask


def shading_map(source, quad, size):
    """Grayscale luminance of the original book region, normalized around 1.0,
    used to multiply onto the new cover so existing shadows/highlights carry over."""
    mask = polygon_mask(quad, size)
    gray = np.asarray(source.convert("L"), dtype=np.float64)
    m = np.asarray(mask, dtype=np.float64) / 255.0
    region = gray[m > 0]
    mean = region.mean() if region.size else 128.0
    shade = gray / mean
    shade = np.clip(shade, 0.5, 1.6)
    return shade


def composite_book(base, cover_path, quad):
    size = base.size
    cover = Image.open(cover_path).convert("RGB")
    warped = warp_into_quad(cover, quad, size)
    mask = polygon_mask(quad, size)

    shade = shading_map(base, quad, size)
    warped_arr = np.asarray(warped.convert("RGB"), dtype=np.float64)
    shaded_arr = np.clip(warped_arr * shade[..., None], 0, 255).astype(np.uint8)
    shaded_img = Image.fromarray(shaded_arr, "RGB").convert("RGBA")

    mask_blurred = mask.filter(ImageFilter.GaussianBlur(1.2))
    shaded_img.putalpha(mask_blurred)
    base.paste(shaded_img, (0, 0), shaded_img)
    return base


def main():
    if len(sys.argv) != 4:
        print("usage: compose.py <left_cover> <center_cover> <right_cover>")
        sys.exit(1)

    base = Image.open(SOURCE).convert("RGBA")
    for name, cover_path in zip(["left", "center", "right"], sys.argv[1:4]):
        base = composite_book(base, cover_path, BOOKS[name])

    base.convert("RGB").save("output.png")
    print("saved output.png")


if __name__ == "__main__":
    main()
