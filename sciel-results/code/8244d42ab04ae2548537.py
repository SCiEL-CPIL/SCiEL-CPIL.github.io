# EVOLVE-BLOCK-START
import numpy as np
import time

_TOL = 1e-7
_N = 10


def _objective(centers, sides):
    return float(np.sum(sides))


def _is_valid(centers, sides, tol=_TOL):
    """Fast vectorized validity check for the public contract."""
    n = len(sides)
    if centers.shape != (n, 2) or sides.shape != (n,):
        return False
    if not (np.all(np.isfinite(centers)) and np.all(np.isfinite(sides))):
        return False
    if np.any(sides <= 0.0):
        return False
    # inside [-tol, 1+tol]^2
    if np.any(centers - sides / 2.0 < -tol) or np.any(centers + sides / 2.0 > 1.0 + tol):
        return False
    # pairwise overlap: forbidden if |dx| < t AND |dy| < t, t=(si+sj)/2 - tol
    cx = centers[:, 0]
    cy = centers[:, 1]
    dx = np.abs(cx[:, None] - cx[None, :])
    dy = np.abs(cy[:, None] - cy[None, :])
    t = (sides[:, None] + sides[None, :]) / 2.0 - tol
    # diagonal is always fine (dx=0,dy=0 -> t>0? t=(si+si)/2-tol=si-tol>0 -> would flag! exclude i==j)
    np.fill_diagonal(dx, np.inf)
    np.fill_diagonal(dy, np.inf)
    if np.any((dx < t) & (dy < t)):
        return False
    return True


def _greedy_enlarge(centers, sides, tol=_TOL, max_iter=400, time_budget=20.0):
    """Deterministically grow each square as much as possible (alternating passes)."""
    n = len(sides)
    t0 = time.time()
    for _ in range(max_iter):
        if time.time() - t0 > time_budget:
            break
        improved = False
        for i in range(n):
            # max side limited by boundary
            s = sides[i]
            cx, cy = centers[i]
            # boundary limits (center fixed)
            lim = min(cx, 1.0 - cx, cy, 1.0 - cy) * 2.0
            s_max = max(s, 0.0)
            # try to increase s
            # new side candidate: start from s, push up until constraint hit
            # use a small multiplicative step for robustness
            for _step in range(60):
                if time.time() - t0 > time_budget:
                    break
                s_try = min(s * 1.002 + 1e-9, lim)
                if s_try <= s + 1e-12:
                    break
                # check overlap with all j != i using new s
                ok = True
                for j in range(n):
                    if j == i:
                        continue
                    t = (s_try + sides[j]) / 2.0 - tol
                    if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                        ok = False
                        break
                if ok:
                    sides[i] = s_try
                    improved = True
                    s = s_try
                else:
                    break
        if not improved:
            break
    # clamp to boundary exactly
    for i in range(n):
        cx, cy = centers[i]
        lim = min(cx, 1.0 - cx, cy, 1.0 - cy) * 2.0
        if sides[i] > lim:
            sides[i] = lim
    return centers, sides


def _seed_heterogeneous():
    """One big square in a corner + a strip of 9 smaller squares filling the rest.
    Layout: big square side B at top-left. Remaining region: to the right (width 1-B, full height)
    and below (width B, height 1-B). We'll just place 9 squares in a grid over the right strip
    and below strip using a simple partition. For a clean feasible start, use:
      - big square side 0.5 at top-left corner: center (0.25,0.75)
      - 9 squares of side 0.5/3? We'll instead tile the L-shape with a 3x3 grid of side (1-B) in the
        right+bottom but that overlaps. Simpler feasible: put 9 squares in a 3x3 grid within the
        square [0.5,1]x[0,1] (right half) -> side (1-0.5)/3 = 1/6.
    """
    B = 0.5
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = (B / 2.0, 1.0 - B / 2.0)  # top-left
    sides[0] = B
    # 9 squares in right half [0.5,1]x[0,1], 3x3 grid
    sub = 1.0 - B  # =0.5
    gs = sub / 3.0
    idx = 1
    for r in range(3):
        for c in range(3):
            cx = B + (c + 0.5) * gs
            cy = (r + 0.5) * gs
            centers[idx] = (cx, cy)
            sides[idx] = gs
            idx += 1
    return centers, sides


def _seed_heterogeneous2():
    """Big square side 0.6 top-left; 9 squares in right strip width 0.4 (3 cols x 3 rows)."""
    B = 0.6
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = (B / 2.0, 1.0 - B / 2.0)
    sides[0] = B
    sub = 1.0 - B  # 0.4
    gs = sub / 3.0
    idx = 1
    for r in range(3):
        for c in range(3):
            cx = B + (c + 0.5) * gs
            cy = (r + 0.5) * gs
            centers[idx] = (cx, cy)
            sides[idx] = gs
            idx += 1
    return centers, sides


def _seed_heterogeneous3():
    """Big square 0.45; 9 squares in right strip width 0.55 (3x3)."""
    B = 0.45
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = (B / 2.0, 1.0 - B / 2.0)
    sides[0] = B
    sub = 1.0 - B
    gs = sub / 3.0
    idx = 1
    for r in range(3):
        for c in range(3):
            cx = B + (c + 0.5) * gs
            cy = (r + 0.5) * gs
            centers[idx] = (cx, cy)
            sides[idx] = gs
            idx += 1
    return centers, sides


def _seed_grid():
    n = 10
    k = int(np.ceil(np.sqrt(n)))
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = np.full(n, side, dtype=np.float64)
    return centers, sides


def squares_in_square10() -> tuple:
    """
    Structured heterogeneous template + greedy feasibility-driven enlargement.
    Returns (centers, sides) per the public contract.
    """
    t_start = time.time()
    deadline = t_start + 600.0 - 30.0  # safety margin

    seeds = [
        _seed_heterogeneous,
        _seed_heterogeneous2,
        _seed_heterogeneous3,
        _seed_grid,
    ]

    best_centers = None
    best_sides = None
    best_val = -np.inf

    for seed_fn in seeds:
        if time.time() > deadline:
            break
        centers, sides = seed_fn()
        # ensure validity of seed (they are constructed feasible, but clamp just in case)
        centers, sides = _greedy_enlarge(centers.copy(), sides.copy(), time_budget=60.0)
        if _is_valid(centers, sides):
            val = _objective(centers, sides)
            if val > best_val:
                best_val = val
                best_centers = centers.copy()
                best_sides = sides.copy()

    if best_centers is None:
        # fallback to grid
        centers, sides = _seed_grid()
        return centers, sides

    return best_centers, best_sides


# EVOLVE-BLOCK-END
