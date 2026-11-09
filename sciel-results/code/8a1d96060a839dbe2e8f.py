# EVOLVE-BLOCK-START
import numpy as np
import time

_TOL = 1e-7
_N = 10
_K = 4  # grid resolution per axis for position search


def _check_validity(centers, sides):
    """Return (is_valid, reason). Mirrors the evaluator's checks."""
    tol = _TOL
    n = len(sides)
    if centers.shape != (n, 2):
        return False, "centers shape"
    if sides.shape != (n,):
        return False, "sides shape"
    if not (np.all(np.isfinite(centers)) and np.all(np.isfinite(sides))):
        return False, "non-finite"
    if not np.all(sides > 0):
        return False, "non-positive side"
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s / 2 < -tol or cx + s / 2 > 1 + tol:
            return False, "x out of bounds"
        if cy - s / 2 < -tol or cy + s / 2 > 1 + tol:
            return False, "y out of bounds"
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2 - tol
            if abs(centers[i, 0] - centers[j, 0]) < t and \
               abs(centers[i, 1] - centers[j, 1]) < t:
                return False, "overlap"
    return True, "ok"


def _max_side_for(centers, sides, i, pos):
    """Largest side for square i if its center were moved to pos, keeping all
    others fixed. Bisection over side length."""
    cx, cy = pos
    n = len(sides)
    # upper bound from walls
    lo = 1e-9
    hi = 2.0 * min(cx, 1.0 - cx, cy, 1.0 - cy) + 2.0 * _TOL
    if hi <= lo:
        return lo
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        ok = True
        for j in range(n):
            if j == i:
                continue
            t = (mid + sides[j]) / 2 - _TOL
            if abs(cx - centers[j, 0]) < t and abs(cy - centers[j, 1]) < t:
                ok = False
                break
        if ok:
            lo = mid
        else:
            hi = mid
    return lo


def _objective(centers, sides):
    """Sum of side lengths (the metric)."""
    return float(np.sum(sides))


def _grid_positions():
    """Dense position grid inside [0,1]^2, including walls and midpoints."""
    pts = np.linspace(0.0, 1.0, _K)
    gx, gy = np.meshgrid(pts, pts, indexing='xy')
    pos = np.stack([gx.ravel(), gy.ravel()], axis=1)
    return pos


def _coordinate_ascent(centers, sides, deadline, max_rounds=60):
    """Iteratively move each square to the best grid position and grow it to its
    max feasible side. Accept only if the total sum improves."""
    pos_grid = _grid_positions()
    n = len(sides)
    best = _objective(centers, sides)
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            cur_cx, cur_cy = centers[i]
            cur_s = sides[i]
            # try every grid position, grow to max feasible side
            best_pos = (cur_cx, cur_cy)
            best_s = cur_s
            for (px, py) in pos_grid:
                s = _max_side_for(centers, sides, i, (px, py))
                if s > best_s + 1e-12:
                    best_s = s
                    best_pos = (px, py)
            if best_s > cur_s + 1e-12:
                # tentative update
                old_c = centers[i].copy()
                old_s = sides[i]
                centers[i] = best_pos
                sides[i] = best_s
                if _check_validity(centers, sides)[0]:
                    new_obj = _objective(centers, sides)
                    if new_obj > best + 1e-12:
                        best = new_obj
                        improved = True
                    else:
                        centers[i] = old_c
                        sides[i] = old_s
                else:
                    centers[i] = old_c
                    sides[i] = old_s
        if not improved:
            break
    return best


def _polish(centers, sides, deadline, rounds=3):
    """High-precision local polish: for each square, fine-tune side via bisection
    around the current value, and nudge positions on a fine grid."""
    fine = np.linspace(0.0, 1.0, 9)
    gx, gy = np.meshgrid(fine, fine, indexing='xy')
    fine_grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
    n = len(sides)
    for _ in range(rounds):
        if time.time() > deadline:
            break
        for i in range(n):
            if time.time() > deadline:
                break
            cur_cx, cur_cy = centers[i]
            cur_s = sides[i]
            # grow current position
            s_here = _max_side_for(centers, sides, i, (cur_cx, cur_cy))
            if s_here > cur_s + 1e-13:
                sides[i] = s_here
            # try fine positions
            best_pos = (cur_cx, cur_cy)
            best_s = sides[i]
            for (px, py) in fine_grid:
                s = _max_side_for(centers, sides, i, (px, py))
                if s > best_s + 1e-13:
                    best_s = s
                    best_pos = (px, py)
            if best_s > sides[i] + 1e-13:
                old_c = centers[i].copy()
                old_s = sides[i]
                centers[i] = best_pos
                sides[i] = best_s
                if _check_validity(centers, sides)[0]:
                    pass
                else:
                    centers[i] = old_c
                    sides[i] = old_s
    return centers, sides


def squares_in_square10() -> tuple:
    """Erdos squares-in-a-square, n=10, maximize sum of side lengths.

    Returns (centers, sides): centers shape (10,2), sides shape (10,).
    """
    start = time.time()
    deadline = start + 600.0 - 30.0  # safety margin

    # Baseline: 4x4 grid, 10 squares.
    n = _N
    k = 4
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = np.full(n, side, dtype=np.float64)

    # Coordinate ascent with bisection growth.
    _coordinate_ascent(centers, sides, deadline, max_rounds=60)

    # High-precision polish.
    centers, sides = _polish(centers, sides, deadline, rounds=3)

    # Final validity guard: if somehow invalid, fall back to the safe baseline.
    if not _check_validity(centers, sides)[0]:
        k = 4
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = np.full(n, side, dtype=np.float64)

    return centers, sides


# EVOLVE-BLOCK-END
