"""Packing 50 equal circles in a unit square [-1/2,1/2]^2.
Directly optimizes the exact min-radius objective with SLSQP."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
HALF = 0.5

def _radius(pts):
    """Exact min radius: min over pairs of half-distance, and over walls."""
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    # pairwise
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    r_pair = 0.5 * d.min()
    # walls
    r_wall = HALF - np.abs(pts).max()
    return min(r_pair, r_wall)

def _neg_radius(pts):
    return -_radius(pts)

def _neg_radius_grad(pts):
    """Gradient of -radius w.r.t. points (100-dim)."""
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    r_pair = 0.5 * d.min()
    r_wall = HALF - np.abs(pts).max()
    r = min(r_pair, r_wall)
    g = np.zeros_like(pts)
    if r_pair <= r_wall:
        # find closest pair
        idx = np.argmin(d)
        i, j = divmod(idx, n)
        v = pts[i] - pts[j]
        norm = np.linalg.norm(v)
        if norm > 0:
            u = v / norm
            g[i] += 0.5 * u
            g[j] -= 0.5 * u
    else:
        # wall binding
        amax = np.abs(pts)
        m = amax.max()
        mask = amax >= m - 1e-12
        # for each binding coordinate, push inward
        for i in range(n):
            for k in range(2):
                if mask[i, k]:
                    # derivative of - (HALF - |x|) w.r.t x = -sign(x)
                    s = np.sign(pts[i, k])
                    if s == 0:
                        s = 1.0
                    g[i, k] -= s
    return g

def _bound_constraints(pts):
    """Penalty-free check that points stay in box (SLSQP box handles this)."""
    pass

def _init_grid(seed=0):
    m = int(np.ceil(np.sqrt(N)))
    rng = np.random.default_rng(seed)
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    pts = grid[:N].astype(np.float64)
    # small jitter to break symmetry
    pts += rng.uniform(-1e-3, 1e-3, size=pts.shape)
    pts = np.clip(pts, -HALF + 1e-6, HALF - 1e-6)
    return pts

def construct():
    t0 = time.time()
    deadline = 135.0  # safety margin before 150s kill
    best_pts = _init_grid(0)
    best_r = _radius(best_pts)

    bounds = [(-HALF, HALF)] * (2 * N)

    # multiple restarts with different seeds
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    for s in seeds:
        if time.time() > deadline - 10.0:
            break
        pts0 = _init_grid(s)
        res = minimize(
            _neg_radius,
            pts0.flatten(),
            jac=_neg_radius_grad,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12},
        )
        if time.time() > deadline - 5.0:
            break
        pts = res.x.reshape(N, 2)
        if np.any(~np.isfinite(pts)):
            continue
        pts = np.clip(pts, -HALF + 1e-9, HALF - 1e-9)
        r = _radius(pts)
        if r > best_r + 1e-12:
            best_r = r
            best_pts = pts.copy()

    # final polish on the best point with tighter tolerance
    if time.time() < deadline - 5.0:
        res2 = minimize(
            _neg_radius,
            best_pts.flatten(),
            jac=_neg_radius_grad,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-15},
        )
        pts2 = res2.x.reshape(N, 2)
        if np.all(np.isfinite(pts2)):
            pts2 = np.clip(pts2, -HALF + 1e-9, HALF - 1e-9)
            r2 = _radius(pts2)
            if r2 > best_r:
                best_pts = pts2

    return best_pts
