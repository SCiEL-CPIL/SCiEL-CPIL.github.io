# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
import itertools

def squares_in_square10() -> tuple:
    """
    Place 10 axis-aligned squares inside the unit square [0,1]^2 to maximize
    the sum of their side lengths.

    Primary construction: 2 large 0.5x0.5 squares + 8 small 0.25x0.25 squares
    arranged in a 4x2 grid in the remaining 0.5x1.0 strip.
    Total = 2*0.5 + 8*0.25 = 3.0

    Then apply local coordinate-ascent refinement to try to improve.
    """
    n = 10
    tol = 1e-7
    deadline = time.time() + 590  # Leave 40s margin for safety

    best_centers = None
    best_sides = None
    best_score = -np.inf

    # --- Primary construction: 2 large + 8 small tiling ---
    centers, sides = construct_2_large_8_small()
    if centers is not None and hard_feasibility(centers, sides):
        best_centers = centers.copy()
        best_sides = sides.copy()
        best_score = float(np.sum(sides))

    # --- Local refinement via coordinate ascent ---
    if best_centers is not None and time.time() < deadline:
        refined_centers, refined_sides, refined_score = coordinate_ascent_refine(
            best_centers, best_sides, deadline)
        if refined_score > best_score:
            best_centers = refined_centers
            best_sides = refined_sides
            best_score = refined_score

    # --- Fallback: uniform 2x5 grid if construction failed ---
    if best_centers is None:
        k1, k2 = 2, 5
        side = 0.2
        cx, cy = np.meshgrid(np.linspace(side/2, 1-side/2, k2),
                             np.linspace(side/2, 1-side/2, k1))
        best_centers = np.column_stack((cx.flatten(), cy.flatten()))
        best_sides = np.full(n, side)
        best_score = float(np.sum(best_sides))

    return best_centers, best_sides


def construct_2_large_8_small():
    """
    Construct the known optimal-ish tiling:
    - Left half [0, 0.5] x [0, 1]: two 0.5x0.5 squares stacked vertically.
    - Right half [0.5, 1] x [0, 1]: eight 0.25x0.25 squares in a 4x2 grid.
    
    Square layout:
    Large squares:
      Square 0: center (0.25, 0.25), side 0.5
      Square 1: center (0.25, 0.75), side 0.5
    Small squares (4 rows x 2 cols in [0.5,1]x[0,1]):
      Row 0 (y in [0, 0.25]): 
        Square 2: center (0.625, 0.125), side 0.25
        Square 3: center (0.875, 0.125), side 0.25
      Row 1 (y in [0.25, 0.5]):
        Square 4: center (0.625, 0.375), side 0.25
        Square 5: center (0.875, 0.375), side 0.25
      Row 2 (y in [0.5, 0.75]):
        Square 6: center (0.625, 0.625), side 0.25
        Square 7: center (0.875, 0.625), side 0.25
      Row 3 (y in [0.75, 1.0]):
        Square 8: center (0.625, 0.875), side 0.25
        Square 9: center (0.875, 0.875), side 0.25
    """
    centers = np.array([
        [0.25, 0.25],
        [0.25, 0.75],
        [0.625, 0.125],
        [0.875, 0.125],
        [0.625, 0.375],
        [0.875, 0.375],
        [0.625, 0.625],
        [0.875, 0.625],
        [0.625, 0.875],
        [0.875, 0.875],
    ])
    sides = np.array([
        0.5, 0.5,
        0.25, 0.25,
        0.25, 0.25,
        0.25, 0.25,
        0.25, 0.25,
    ])
    return centers, sides


def coordinate_ascent_refine(centers, sides, deadline):
    """
    Local coordinate-ascent refinement:
    - Iterate over each square, adjusting its center and side to maximize
      total sum while maintaining feasibility.
    - Use small perturbations and accept only if feasible and improves.
    """
    n = len(centers)
    tol = 1e-7
    centers = centers.copy().astype(float)
    sides = sides.copy().astype(float)
    
    # Perturbation schedule
    step_sizes = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    
    best_score = float(np.sum(sides))
    
    for step in step_sizes:
        if time.time() > deadline:
            break
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    break
                # Try perturbing center x, center y, and side
                for dx, dy, ds in [
                    (step, 0, 0), (-step, 0, 0),
                    (0, step, 0), (0, -step, 0),
                    (0, 0, step), (0, 0, -step),
                    (step, step, 0), (-step, -step, 0),
                    (step, -step, 0), (-step, step, 0),
                    (0, 0, step*0.5), (0, 0, -step*0.5),
                ]:
                    new_cx = centers[i, 0] + dx
                    new_cy = centers[i, 1] + dy
                    new_s = sides[i] + ds
                    if new_s <= 0:
                        continue
                    # Quick boundary check
                    if new_cx - new_s/2 < -tol or new_cx + new_s/2 > 1 + tol:
                        continue
                    if new_cy - new_s/2 < -tol or new_cy + new_s/2 > 1 + tol:
                        continue
                    # Try this perturbation
                    old_cx, old_cy, old_s = centers[i, 0], centers[i, 1], sides[i]
                    centers[i, 0] = new_cx
                    centers[i, 1] = new_cy
                    sides[i] = new_s
                    if hard_feasibility(centers, sides):
                        new_score = float(np.sum(sides))
                        if new_score > best_score + 1e-12:
                            best_score = new_score
                            improved = True
                        else:
                            # Revert
                            centers[i, 0] = old_cx
                            centers[i, 1] = old_cy
                            sides[i] = old_s
                    else:
                        # Revert
                        centers[i, 0] = old_cx
                        centers[i, 1] = old_cy
                        sides[i] = old_s
    
    return centers, sides, best_score

def hard_feasibility(centers, sides):
    """Strict check for the final candidate."""
    n = len(centers)
    tol = 1e-7
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if not np.isfinite(cx) or not np.isfinite(cy) or not np.isfinite(s):
            return False
        if s <= 0:
            return False
        if cx - s / 2 < -tol or cx + s / 2 > 1 + tol:
            return False
        if cy - s / 2 < -tol or cy + s / 2 > 1 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2 - tol
            if abs(centers[i][0] - centers[j][0]) < t and abs(centers[i][1] - centers[j][1]) < t:
                return False
    return True

# EVOLVE-BLOCK-END
