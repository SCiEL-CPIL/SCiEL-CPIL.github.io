"""Rewritten program for circle packing in perimeter-4 rectangle."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _pack_rectangular_grid(n_circles, w, h):
    """
    Pack n_circles into a rectangle of width w and height h using a grid layout.
    Returns an array of shape (n_circles, 3) with (cx, cy, r).
    """
    # Determine grid dimensions
    # Try different splits to find the one that maximizes minimum radius
    best_r = 0.0
    best_positions = None
    best_radii = None

    for cols in range(1, n_circles + 1):
        rows = (n_circles + cols - 1) // cols
        # Check if this grid can fit
        r_x = w / (2 * cols)
        r_y = h / (2 * rows)
        r = min(r_x, r_y)
        if r > best_r:
            best_r = r
            # Generate positions
            positions = []
            radii = []
            for i in range(cols):
                for j in range(rows):
                    if len(positions) >= n_circles:
                        break
                    cx = (i + 0.5) * w / cols
                    cy = (j + 0.5) * h / rows
                    positions.append((cx, cy))
                    radii.append(r)
            best_positions = positions
            best_radii = radii
        if len(best_positions) and len(best_positions) >= n_circles:
            break

    if best_positions is None:
        # Fallback: single row
        cols = n_circles
        rows = 1
        r = min(w / (2 * cols), h / 2)
        positions = []
        radii = []
        for i in range(cols):
            cx = (i + 0.5) * w / cols
            cy = h / 2
            positions.append((cx, cy))
            radii.append(r)
        best_positions = positions
        best_radii = radii

    circles = np.array([(p[0], p[1], r) for p, r in zip(best_positions[:n_circles], best_radii[:n_circles])])
    return circles


def _validate(circles, w, h, tol=1e-9):
    """
    Check if circles are valid:
    - All centers within [0, w] x [0, h]
    - All radii positive
    - No overlapping circles
    - All circles fit within the rectangle
    """
    n = len(circles)
    if n == 0:
        return False

    # Check bounds and positive radii
    for i in range(n):
        cx, cy, r = circles[i]
        if r <= 0:
            return False
        if cx - r < -tol or cx + r > w + tol:
            return False
        if cy - r < -tol or cy + r > h + tol:
            return False

    # Check pairwise distances
    for i in range(n):
        for j in range(i + 1, n):
            cx1, cy1, r1 = circles[i]
            cx2, cy2, r2 = circles[j]
            dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
            if dist < r1 + r2 - tol:
                return False

    return True


def _sum_radii(circles):
    """Compute the sum of radii."""
    return np.sum(circles[:, 2])


def construct():
    """
    Pack 21 circles in a rectangle of perimeter 4 (i.e., 2*(w+h) = 4, so w+h = 2).
    Maximize the sum of radii.
    Returns an array of shape (21, 3) with (cx, cy, r).
    """
    n_circles = N

    # Since w + h = 2, let's try different aspect ratios.
    # For a grid packing, the optimal aspect ratio depends on the grid dimensions.
    # We'll try several aspect ratios and pick the best one.

    best_circles = None
    best_sum = -np.inf

    # Try different aspect ratios w/h
    # w + h = 2, so h = 2 - w
    # w ranges from small to large
    aspect_ratios = np.linspace(0.1, 10.0, 50)
    for ar in aspect_ratios:
        w = 2.0 * ar / (1 + ar)
        h = 2.0 - w
        if w <= 0 or h <= 0:
            continue

        circles = _pack_rectangular_grid(n_circles, w, h)
        if _validate(circles, w, h):
            s = _sum_radii(circles)
            if s > best_sum:
                best_sum = s
                best_circles = circles.copy()

    # Also try a few specific aspect ratios that might be optimal for common grid layouts
    # For 21 circles, possible grids: 3x7, 7x3, 4x6 (with one missing), etc.
    specific_grids = [
        (3, 7),  # 3 cols, 7 rows
        (7, 3),  # 7 cols, 3 rows
        (4, 6),  # 4 cols, 6 rows (24 slots, use 21)
        (6, 4),  # 6 cols, 4 rows
        (5, 5),  # 5 cols, 5 rows (25 slots, use 21)
        (1, 21),
        (21, 1),
    ]
    for cols, rows in specific_grids:
        # For a grid with cols and rows, the optimal aspect ratio is w/h = (cols * r) / (rows * r) = cols/rows
        # But since circles have equal radius in a grid, w = cols * 2r, h = rows * 2r
        # So w/h = cols/rows
        ar = cols / rows
        w = 2.0 * ar / (1 + ar)
        h = 2.0 - w
        if w <= 0 or h <= 0:
            continue

        circles = _pack_rectangular_grid(n_circles, w, h)
        if _validate(circles, w, h):
            s = _sum_radii(circles)
            if s > best_sum:
                best_sum = s
                best_circles = circles.copy()

    if best_circles is None:
        # Fallback: default grid
        best_circles = _pack_rectangular_grid(n_circles, 1.0, 1.0)

    return best_circles
# EVOLVE-BLOCK-END
