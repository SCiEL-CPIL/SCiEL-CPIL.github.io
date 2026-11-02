"""Model-and-optimize approach for the Heilbronn-11 triangle problem."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import itertools

S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # area of the equilateral triangle with side 1

DEADLINE = 142  # safety margin from 150s limit

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])


def _project_to_triangle(pts):
    """Vectorized projection of points into the unit equilateral triangle."""
    y = np.clip(pts[:, 1], 0.0, S3H)
    x_min = y / S3
    x_max = 1.0 - y / S3
    x = np.clip(pts[:, 0], x_min, x_max)
    return np.column_stack([x, y])


def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points (vectorized)."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return np.min(areas)


def _soft_min_area(pts, beta):
    """Soft minimum of triangle areas using log-sum-exp (differentiable)."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = max_a - log_sum / beta
    return soft_min


def _barycentric_to_cartesian(bary):
    """Convert barycentric coordinates (u, v, w) with u+v+w=1 to Cartesian."""
    # bary has shape (n, 3) or we work with (n, 2) where w = 1-u-v
    u = bary[:, 0]
    v = bary[:, 1]
    x = u * 1.0 + v * 0.5
    y = v * S3H
    return np.column_stack([x, y])


def _cartesian_to_barycentric(pts):
    """Convert Cartesian to barycentric (u, v) where w = 1-u-v >= 0."""
    x = pts[:, 0]
    y = pts[:, 1]
    v = y / S3H
    u = x - 0.5 * v
    return np.column_stack([u, v])


def construct():
    start_time = time.time()
    rng = np.random.default_rng(12345)

    best_pts = None
    best_area = 0.0

    def _elapsed_ok(margin):
        return (time.time() - start_time) < (DEADLINE - margin)

    def _update_best(pts):
        nonlocal best_pts, best_area
        area = _min_triangle_area(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()

    # ---- Model: initialize a plausible parameterization (corners, mid-edges, centroid) ----
    init = np.array([
        [0.0, 0.0],          # corner 0
        [1.0, 0.0],          # corner 1
        [0.5, S3H],          # corner 2
        [0.5, 0.0],          # mid bottom
        [0.75, S3H / 2],     # mid right
        [0.25, S3H / 2],     # mid left
        [1/3, S3H/3],        # centroid
        [0.333, 0.0],
        [0.667, 0.0],
        [0.5, S3H/3],
        [0.2, 0.1],
    ], dtype=float)
    _update_best(init)

    def _start_from_base(base, scale):
        pts = base + rng.normal(0, scale, size=(N, 2))
        return _project_to_triangle(pts)

    def _optimize_from(pts0, beta):
        """Optimize in barycentric coordinates with L-BFGS-B (naturally inside triangle)."""
        # Convert to barycentric (u, v) where w = 1-u-v
        bary0 = _cartesian_to_barycentric(pts0)
        
        def objective(x):
            u = x[0:N]
            v = x[N:2*N]
            # Clip to valid barycentric range: u>=0, v>=0, u+v<=1
            u = np.clip(u, 0, 1)
            v = np.clip(v, 0, 1 - u)
            pts = _barycentric_to_cartesian(np.column_stack([u, v]))
            return -_soft_min_area(pts, beta)
        
        try:
            x0 = np.concatenate([bary0[:, 0], bary0[:, 1]])
            res = minimize(objective, x0, method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12})
            u = np.clip(res.x[0:N], 0, 1)
            v = np.clip(res.x[N:2*N], 0, 1 - u)
            pts = _barycentric_to_cartesian(np.column_stack([u, v]))
            return pts
        except Exception:
            return pts0

    # ---- Phase A: multi-start soft-min gradient optimization (annealed beta) ----
    n_starts = 36
    bases = [init]
    for _ in range(18):
        bases.append(_start_from_base(init, 0.25))

    for s in range(n_starts):
        if not _elapsed_ok(30):
            break
        base = bases[s % len(bases)]
        pts = _start_from_base(base, 0.05)
        _update_best(pts)
        for beta in [10.0, 30.0, 80.0, 200.0, 500.0, 1200.0]:
            if not _elapsed_ok(30):
                break
            pts = _optimize_from(pts, beta)
            _update_best(pts)

    # ---- Phase B: Nelder-Mead local polish on the best configuration ----
    if best_pts is not None and _elapsed_ok(12):
        for scale in [0.04, 0.02, 0.01, 0.005]:
            if not _elapsed_ok(8):
                break
            for _ in range(8):
                if not _elapsed_ok(8):
                    break
                cand = best_pts + rng.normal(0, scale, size=(N, 2))
                cand = _project_to_triangle(cand)

                def nm_obj(x):
                    p = x.reshape(-1, 2)
                    p = _project_to_triangle(p)
                    return -_soft_min_area(p, 800.0)
                try:
                    res = minimize(nm_obj, cand.flatten(), method='Nelder-Mead',
                                   options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
                    p = _project_to_triangle(res.x.reshape(-1, 2))
                    _update_best(p)
                except Exception:
                    pass

    # ---- Final fine polish on the absolute best ----
    if best_pts is not None and _elapsed_ok(6):
        def nm_obj(x):
            p = x.reshape(-1, 2)
            p = _project_to_triangle(p)
            return -_soft_min_area(p, 2000.0)
        try:
            res = minimize(nm_obj, best_pts.flatten(), method='Nelder-Mead',
                           options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14})
            p = _project_to_triangle(res.x.reshape(-1, 2))
            _update_best(p)
        except Exception:
            pass

    if best_pts is None:
        # fallback random feasible configuration
        u = rng.random(N)
        v = rng.random(N)
        mask = u + v > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        pts = _barycentric_to_cartesian(np.column_stack([u, v]))
        return pts

    # Final projection to ensure validity
    best_pts = _project_to_triangle(best_pts)
    return best_pts
# EVOLVE-BLOCK-END
