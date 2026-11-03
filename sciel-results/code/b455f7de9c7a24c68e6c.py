"""Pack 200 equal circles in a unit square to maximize common radius."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, cdist
import time

N = 200
_TIME_LIMIT = 110  # seconds, leave margin before 120s kill


def _score(points):
    """Compute the radius for a set of points in [-0.5, 0.5]^2."""
    if not np.all(np.isfinite(points)):
        return 0.0
    if not np.all((points >= -0.5) & (points <= 0.5)):
        return 0.0
    if len(points) < 2:
        min_pair = float('inf')
    else:
        d = pdist(points)
        min_pair = np.min(d)
    dist_boundary = 0.5 - np.max(np.abs(points), axis=1)
    min_boundary = np.min(dist_boundary)
    return min(min_pair / 2.0, min_boundary)


def _best_result():
    """Fallback: 10x20 rectangular grid."""
    rows, cols = 20, 10
    dx = 1.0 / cols
    dy = 1.0 / rows
    xs = -0.5 + (np.arange(cols) + 0.5) * dx
    ys = -0.5 + (np.arange(rows) + 0.5) * dy
    grid = np.array([(x, y) for y in ys for x in xs])
    return grid[:N]


def _rect_grid_score(params, rows, cols):
    """Score (radius) for a rectangular grid. params = [dx, dy, x0, y0]."""
    dx, dy, x0, y0 = params
    if dx <= 0 or dy <= 0:
        return -1e10
    xs = x0 + np.arange(cols) * dx
    ys = y0 + np.arange(rows) * dy
    pts = np.array([(x, y) for y in ys for x in xs])[:N]
    if len(pts) < N:
        return -1e10
    if np.any(pts[:, 0] < -0.5) or np.any(pts[:, 0] > 0.5) or \
       np.any(pts[:, 1] < -0.5) or np.any(pts[:, 1] > 0.5):
        return -1e10
    wall = np.min([
        np.min(pts[:, 0]) + 0.5,
        0.5 - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + 0.5,
        0.5 - np.max(pts[:, 1]),
    ])
    if N > 1:
        D = cdist(pts, pts)
        np.fill_diagonal(D, np.inf)
        min_pair = np.min(D)
    else:
        min_pair = np.inf
    r = min(wall, min_pair / 2)
    if r <= 0:
        return -1e10
    return r


def _hex_grid_score(params, rows, cols):
    """Score for a hexagonal (staggered) grid.
    params = [dx, dy, x0, y0, offset]
    """
    dx, dy, x0, y0, off = params
    if dx <= 0 or dy <= 0:
        return -1e10
    pts = []
    for i in range(rows):
        y = y0 + i * dy
        shift = off * dx if i % 2 == 1 else 0.0
        xs = x0 + shift + np.arange(cols) * dx
        for x in xs:
            pts.append((x, y))
    pts = np.array(pts[:N])
    if len(pts) < N:
        return -1e10
    if np.any(pts[:, 0] < -0.5) or np.any(pts[:, 0] > 0.5) or \
       np.any(pts[:, 1] < -0.5) or np.any(pts[:, 1] > 0.5):
        return -1e10
    wall = np.min([
        np.min(pts[:, 0]) + 0.5,
        0.5 - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + 0.5,
        0.5 - np.max(pts[:, 1]),
    ])
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D)
    r = min(wall, min_pair / 2)
    if r <= 0:
        return -1e10
    return r


def _neg_rect_score(params, rows, cols):
    return -_rect_grid_score(params, rows, cols)


def _neg_hex_score(params, rows, cols):
    return -_hex_grid_score(params, rows, cols)


def _local_refine(points, time_budget):
    """Local refinement using scipy minimize on the negative radius score."""
    def _neg_score(pts):
        pts = pts.reshape(-1, 2)
        if not np.all((pts >= -0.5) & (pts <= 0.5)):
            return 1e10
        if not np.all(np.isfinite(pts)):
            return 1e10
        r = _score(pts)
        return -r

    start = time.time()
    current = points.copy()
    current_score = _score(current)
    
    # Use L-BFGS-B with bounds
    bounds = [(-0.5, 0.5)] * (2 * N)
    x0 = current.flatten()
    
    try:
        res = minimize(_neg_score, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
        if time.time() - start < time_budget:
            new_pts = res.x.reshape(-1, 2)
            new_score = _score(new_pts)
            if new_score > current_score:
                current = new_pts
                current_score = new_score
    except Exception:
        pass
    
    # Additional Nelder-Mead refinement if time remains
    remaining = time_budget - (time.time() - start)
    if remaining > 10:
        try:
            res2 = minimize(_neg_score, current.flatten(), method='Nelder-Mead',
                            options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12})
            new_pts = res2.x.reshape(-1, 2)
            new_score = _score(new_pts)
            if new_score > current_score:
                current = new_pts
                current_score = new_score
        except Exception:
            pass
    
    return current, current_score


def construct():
    t0 = time.time()
    best_r = 0.0
    best_pts = _best_result()
    best_r = _score(best_pts)

    # Try rectangular grids with various row/col splits
    factor_pairs = []
    for c in range(1, int(np.sqrt(N)) + 1):
        if N % c == 0:
            r = N // c
            factor_pairs.append((r, c))
            factor_pairs.append((c, r))
    factor_pairs = list(set(factor_pairs))

    extra = [(14, 15), (15, 14), (20, 10), (10, 20), (25, 8), (8, 25),
             (16, 13), (13, 16), (12, 17), (17, 12)]
    for pair in extra:
        if pair not in factor_pairs:
            factor_pairs.append(pair)

    # Rectangular grids
    for (rows, cols) in factor_pairs:
        if time.time() - t0 > _TIME_LIMIT:
            break
        dx0 = 1.0 / cols
        dy0 = 1.0 / rows
        x00 = -0.5 + dx0 / 2
        y00 = -0.5 + dy0 / 2
        x0 = np.array([dx0, dy0, x00, y00])
        try:
            res = minimize(_neg_rect_score, x0, args=(rows, cols), method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
            r = -res.fun
            if r > best_r:
                best_r = r
                dx, dy, x0v, y0v = res.x
                xs = x0v + np.arange(cols) * dx
                ys = y0v + np.arange(rows) * dy
                pts = np.array([(x, y) for y in ys for x in xs])[:N]
                if np.all(pts[:, 0] >= -0.5) and np.all(pts[:, 0] <= 0.5) and \
                   np.all(pts[:, 1] >= -0.5) and np.all(pts[:, 1] <= 0.5):
                    best_pts = pts
        except Exception:
            pass

    # Hexagonal grids
    for (rows, cols) in factor_pairs:
        if time.time() - t0 > _TIME_LIMIT:
            break
        dx0 = 1.0 / cols
        dy0 = dx0 * np.sqrt(3) / 2
        if (rows - 1) * dy0 > 1.0:
            dy0 = 1.0 / (rows - 1) if rows > 1 else 1.0
        x00 = -0.5 + dx0 / 2
        y00 = -0.5 + dy0 / 2
        off0 = 0.5
        x0 = np.array([dx0, dy0, x00, y00, off0])
        try:
            res = minimize(_neg_hex_score, x0, args=(rows, cols), method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
            r = -res.fun
            if r > best_r:
                best_r = r
                dx, dy, x0v, y0v, offv = res.x
                pts = []
                for i in range(rows):
                    y = y0v + i * dy
                    shift = offv * dx if i % 2 == 1 else 0.0
                    xs = x0v + shift + np.arange(cols) * dx
                    for x in xs:
                        pts.append((x, y))
                pts = np.array(pts[:N])
                if len(pts) == N and \
                   np.all(pts[:, 0] >= -0.5) and np.all(pts[:, 0] <= 0.5) and \
                   np.all(pts[:, 1] >= -0.5) and np.all(pts[:, 1] <= 0.5):
                    best_pts = pts
        except Exception:
            pass

    # Local refinement of the best configuration
    remaining = _TIME_LIMIT - (time.time() - t0)
    if remaining > 5:
        best_pts, best_r = _local_refine(best_pts, remaining - 5)

    if best_r <= 0:
        return _best_result()
    return best_pts
