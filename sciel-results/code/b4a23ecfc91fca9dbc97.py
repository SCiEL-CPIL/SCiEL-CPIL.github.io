"""Heilbronn triangles: maximize the smallest triangle area among 11 points."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import itertools

S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4

DEADLINE = 140.0  # Safety margin from 150s limit

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
    min_a = np.min(areas)
    shifted = -beta * (areas - min_a)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = min_a - log_sum / beta
    return soft_min

def _barycentric_to_cartesian(bary):
    """Convert barycentric coordinates (u, v) where w=1-u-v to Cartesian."""
    u = bary[:, 0]
    v = bary[:, 1]
    x = u + v * 0.5
    y = v * S3H
    return np.column_stack([x, y])

def _cartesian_to_barycentric(pts):
    """Convert Cartesian to barycentric (u, v) where w = 1-u-v >= 0."""
    x = pts[:, 0]
    y = pts[:, 1]
    v = y / S3H
    u = x - 0.5 * v
    return np.column_stack([u, v])

def _random_bary(rng):
    """Generate random barycentric coordinates (u, v) with u+v<=1."""
    u = rng.random(N)
    v = rng.random(N)
    mask = u + v > 1
    u[mask] = 1 - u[mask]
    v[mask] = 1 - v[mask]
    return np.column_stack([u, v])

def _bary_objective(flat_bary, beta):
    """Objective function in barycentric coordinates for L-BFGS-B."""
    u = flat_bary[0:N]
    v = flat_bary[N:2*N]
    # Clip to valid barycentric range
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1 - u)
    bary = np.column_stack([u, v])
    pts = _barycentric_to_cartesian(bary)
    return -_soft_min_area(pts, beta)

def _optimize_bary(pts0, beta, maxiter=500):
    """Optimize in barycentric coordinates using L-BFGS-B."""
    bary0 = _cartesian_to_barycentric(pts0)
    x0 = np.concatenate([bary0[:, 0], bary0[:, 1]])
    
    try:
        res = minimize(_bary_objective, x0, method='L-BFGS-B',
                       args=(beta,),
                       options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-12})
        u = np.clip(res.x[0:N], 0, 1)
        v = np.clip(res.x[N:2*N], 0, 1 - u)
        bary = np.column_stack([u, v])
        return _barycentric_to_cartesian(bary)
    except Exception:
        return pts0

def _nm_polish(pts0, beta=800, maxiter=5000):
    """Nelder-Mead local polish in Cartesian space."""
    def nm_obj(x):
        p = x.reshape(-1, 2)
        p = _project_to_triangle(p)
        return -_soft_min_area(p, beta)
    
    try:
        res = minimize(nm_obj, pts0.flatten(), method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12})
        return _project_to_triangle(res.x.reshape(-1, 2))
    except Exception:
        return pts0

def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_area = 0.0
    
    def _elapsed_ok(margin):
        return (time.time() - start_time) < (DEADLINE - margin)
    
    def _update_best(pts):
        nonlocal best_pts, best_area
        pts = _project_to_triangle(pts)
        area = _min_triangle_area(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()
    
    # Structured initial configurations
    init_configs = [
        # Corners + midpoints + centroid
        np.array([
            [0.0, 0.0], [1.0, 0.0], [0.5, S3H],
            [0.5, 0.0], [0.75, S3H/2], [0.25, S3H/2],
            [1/3, S3H/3], [0.333, 0.0], [0.667, 0.0],
            [0.5, S3H/3], [0.2, 0.1]
        ]),
        # Symmetric grid-like
        np.array([
            [0.1, 0.05], [0.5, 0.05], [0.9, 0.05],
            [0.2, S3H/3], [0.5, S3H/3], [0.8, S3H/3],
            [0.3, 2*S3H/3], [0.5, 2*S3H/3], [0.7, 2*S3H/3],
            [0.4, 0.75*S3H], [0.6, 0.75*S3H]
        ]),
        # Concentric triangles
        np.array([
            [0.1, 0.05], [0.5, 0.05], [0.9, 0.05],
            [0.25, S3H/2], [0.5, S3H/2], [0.75, S3H/2],
            [0.4, 0.7*S3H], [0.5, 0.7*S3H], [0.6, 0.7*S3H],
            [0.45, 0.85*S3H], [0.55, 0.85*S3H]
        ]),
    ]
    
    for cfg in init_configs:
        _update_best(cfg)
    
    # Generate diverse starting points
    starts = []
    for cfg in init_configs:
        for scale in [0.05, 0.1, 0.2]:
            starts.append(_project_to_triangle(cfg + rng.normal(0, scale, size=(N, 2))))
    for _ in range(15):
        starts.append(_barycentric_to_cartesian(_random_bary(rng)))
    
    # Annealed L-BFGS-B optimization with multiple starts
    betas = [10.0, 30.0, 80.0, 200.0, 500.0, 1000.0]
    
    for s, start_pts in enumerate(starts):
        if not _elapsed_ok(20):
            break
        
        _update_best(start_pts)
        
        current_pts = start_pts.copy()
        for beta in betas:
            if not _elapsed_ok(15):
                break
            current_pts = _optimize_bary(current_pts, beta, maxiter=300)
            _update_best(current_pts)
    
    # Nelder-Mead polish on best configuration
    if best_pts is not None and _elapsed_ok(10):
        for scale in [0.03, 0.015, 0.008]:
            if not _elapsed_ok(8):
                break
            for _ in range(5):
                if not _elapsed_ok(6):
                    break
                cand = best_pts + rng.normal(0, scale, size=(N, 2))
                cand = _project_to_triangle(cand)
                polished = _nm_polish(cand, beta=1000, maxiter=3000)
                _update_best(polished)
    
    # Final high-precision polish
    if best_pts is not None and _elapsed_ok(5):
        final = _nm_polish(best_pts, beta=2000, maxiter=10000)
        _update_best(final)
    
    if best_pts is None:
        # Fallback
        bary = _random_bary(rng)
        best_pts = _barycentric_to_cartesian(bary)
    
    return _project_to_triangle(best_pts)
# EVOLVE-BLOCK-END
