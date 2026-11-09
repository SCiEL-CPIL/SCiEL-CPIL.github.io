# EVOLVE-BLOCK-START
import time
import numpy as np

_TOL = 1e-7
_DEADLINE = 630.0
_MARGIN = 15.0


def _penalty(centers, sides, w=200.0):
    """Penalty = overlap penalty + boundary penalty. Lower is better."""
    n = len(sides)
    cx, cy = centers[:, 0], centers[:, 1]
    s = sides
    pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            t = 0.5 * (s[i] + s[j]) - _TOL
            dx = np.abs(cx[i] - cx[j])
            dy = np.abs(cy[i] - cy[j])
            gap = np.minimum(dx, dy) - t
            if gap < 0.0:
                pen += w * gap * gap
    # boundary: keep each square inside [-tol, 1+tol]^2
    lo_x = cx - 0.5 * s
    hi_x = cx + 0.5 * s
    lo_y = cy - 0.5 * s
    hi_y = cy + 0.5 * s
    pen += w * np.maximum(0.0, -lo_x + _TOL) ** 2
    pen += w * np.maximum(0.0, hi_x - (1.0 + _TOL)) ** 2
    pen += w * np.maximum(0.0, -lo_y + _TOL) ** 2
    pen += w * np.maximum(0.0, hi_y - (1.0 + _TOL)) ** 2
    return pen


def _valid(centers, sides):
    n = len(sides)
    cx, cy = centers[:, 0], centers[:, 1]
    s = sides
    if not (np.all(np.isfinite(centers)) and np.all(np.isfinite(s))):
        return False
    if np.any(s <= 0.0):
        return False
    if np.any(cx - 0.5 * s < -_TOL) or np.any(cx + 0.5 * s > 1.0 + _TOL):
        return False
    if np.any(cy - 0.5 * s < -_TOL) or np.any(cy + 0.5 * s > 1.0 + _TOL):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            t = 0.5 * (s[i] + s[j]) - _TOL
            if np.abs(cx[i] - cx[j]) < t and np.abs(cy[i] - cy[j]) < t:
                return False
    return True


def _obj(centers, sides):
    if _valid(centers, sides):
        return -np.sum(sides)
    return np.sum(sides) + 1e6 + 10.0 * _penalty(centers, sides)


def _init_packing():
    """Structural 0.5/0.25 split: 1 big square (0.5) + 4 medium (0.25) + 5 small (0.2)."""
    centers = np.zeros((10, 2))
    sides = np.zeros(10)
    # 1 big square (0.5) in top-left
    centers[0] = [0.25, 0.75]
    sides[0] = 0.5
    # 4 medium squares (0.25) in bottom-left 0.5x0.5 region
    centers[1] = [0.125, 0.125]
    centers[2] = [0.375, 0.125]
    centers[3] = [0.125, 0.375]
    centers[4] = [0.375, 0.375]
    sides[1:5] = 0.25
    # 5 small squares (0.2) in the right 0.5-wide column
    centers[5] = [0.75, 0.10]
    centers[6] = [0.75, 0.30]
    centers[7] = [0.75, 0.50]
    centers[8] = [0.75, 0.70]
    centers[9] = [0.75, 0.90]
    sides[5:10] = 0.2
    return centers, sides


def _lbfgs(centers, sides, time_limit):
    """Coordinate-free smooth optimization via L-BFGS-B on concatenated params."""
    x0 = np.concatenate([centers.ravel(), sides])
    best = (centers.copy(), sides.copy(), _obj(centers, sides))

    def fun(x):
        c = x[:20].reshape(10, 2)
        s = x[20:]
        return _obj(c, s)

    try:
        from scipy.optimize import minimize
        res = minimize(fun, x0, method='L-BFGS-B',
                       options={'maxiter': 600, 'maxfun': 6000, 'maxls': 40})
        if res.x is not None:
            c = res.x[:20].reshape(10, 2)
            s = res.x[20:]
            o = _obj(c, s)
            if o < best[2]:
                best = (c, s, o)
    except Exception:
        pass
    return best


def squares_in_square10() -> tuple:
    t0 = time.time()
    # Baseline: uniform grid (feasible, sum = 10/3 ≈ 3.333)
    n = 10
    k = int(np.ceil(np.sqrt(n)))
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    grid_centers = np.stack([cx, cy], axis=1).astype(np.float64)
    grid_sides = np.full(n, side, dtype=np.float64)
    best = (grid_centers, grid_sides, _obj(grid_centers, grid_sides))

    # Structural init + optimization
    c0, s0 = _init_packing()
    c0, s0 = c0.astype(np.float64), s0.astype(np.float64)
    if time.time() - t0 < _DEADLINE - _MARGIN:
        c, s, o = _lbfgs(c0, s0, _DEADLINE - _MARGIN)
        if o < best[2]:
            best = (c, s, o)

    # A few random restarts from perturbed structural init if time permits
    rng = np.random.default_rng(12345)
    for _ in range(40):
        if time.time() - t0 > _DEADLINE - _MARGIN:
            break
        c = c0 + rng.normal(0, 0.02, size=c0.shape)
        s = s0 * (1.0 + rng.normal(0, 0.02, size=s0.shape))
        c, s, o = _lbfgs(c, s, _DEADLINE - _MARGIN)
        if o < best[2]:
            best = (c, s, o)

    centers, sides, _ = best
    # Final validity guard: fall back to grid if best is somehow invalid
    if not _valid(centers, sides):
        centers, sides = grid_centers, grid_sides
    return centers, sides


# EVOLVE-BLOCK-END
