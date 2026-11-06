"""Optimized 8-point set maximizing product of distances at fixed diameter."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

_N = 8
_DEADLINE = 150.0
_SAFETY = 20.0
_START = time.time()

def _remaining():
    return _DEADLINE - _SAFETY - (time.time() - _START)

def _diameter(pts):
    """Compute the diameter (max pairwise distance) of a point set."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    return np.max(dists)

def _log_product(pts):
    """Compute log of product of all ordered pair distances (i != j)."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    mask = ~np.eye(n, dtype=bool)
    return np.sum(np.log(dists[mask] + 1e-300))

def _score_from_log_product_and_diameter(log_prod, diam):
    """Compute Delta/n^n given log product and diameter."""
    if diam < 1e-15:
        return 0.0
    scale = 2.0 / diam
    # log of scaled product = log_prod + 56 * log(scale)
    # since there are 8*7 = 56 ordered pairs
    log_scaled_prod = log_prod + _N * (_N - 1) * math.log(scale)
    return math.exp(log_scaled_prod / (_N ** _N))

def _score(points):
    """Compute Delta/n^n where Delta is product of all ordered pair distances,
    after scaling so diameter = 2."""
    pts = np.array(points, dtype=np.float64)
    diam = _diameter(pts)
    if diam < 1e-15:
        return 0.0
    log_prod = _log_product(pts)
    return _score_from_log_product_and_diameter(log_prod, diam)

def _objective(x):
    """Negative log score for minimization."""
    if _remaining() < 5:
        return 1e10
    pts = x.reshape(_N, 2)
    diam = _diameter(pts)
    if diam < 1e-15:
        return 1e10
    log_prod = _log_product(pts)
    log_scaled_prod = log_prod + _N * (_N - 1) * math.log(2.0 / diam)
    return -log_scaled_prod / (_N ** _N)

def _make_cage_init(rng, style=0):
    """Create diverse initial configurations."""
    n = _N
    if style == 0:
        # Regular octagon
        angles = 2 * np.pi * np.arange(n) / n
        return np.column_stack([np.cos(angles), np.sin(angles)])
    elif style == 1:
        # Perturbed regular octagon
        angles = 2 * np.pi * np.arange(n) / n + rng.normal(0, 0.15, n)
        angles = np.sort(angles)
        return np.column_stack([np.cos(angles), np.sin(angles)])
    elif style == 2:
        # Two squares
        a1 = 2 * np.pi * np.arange(4) / 4
        a2 = a1 + np.pi / 4
        return np.vstack([
            np.column_stack([np.cos(a1), np.sin(a1)]),
            np.column_stack([np.cos(a2), np.sin(a2)])
        ])
    elif style == 3:
        # Rectangle-like: 4 on top, 4 on bottom
        x_top = np.array([-1.2, -0.4, 0.4, 1.2])
        x_bot = np.array([-1.2, -0.4, 0.4, 1.2])
        return np.column_stack([np.concatenate([x_top, x_bot]),
                                np.concatenate([np.ones(4) * 0.8, -np.ones(4) * 0.8])])
    elif style == 4:
        # Heptagon + center
        angles = 2 * np.pi * np.arange(7) / 7
        pts7 = np.column_stack([np.cos(angles), np.sin(angles)])
        return np.vstack([pts7, [[0.0, 0.0]]])
    elif style == 5:
        # Random points in unit disk
        r = rng.uniform(0.3, 1.0, n)
        theta = rng.uniform(0, 2 * np.pi, n)
        return np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    else:
        # Random
        return rng.uniform(-1, 1, (n, 2))

def _refine(x0, method='L-BFGS-B', maxiter=5000):
    """Refine a configuration using the specified optimizer."""
    if _remaining() < 5:
        return x0, _score(x0.reshape(_N, 2))
    try:
        if method == 'L-BFGS-B':
            res = minimize(_objective, x0, method='L-BFGS-B',
                          options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12})
        elif method == 'Nelder-Mead':
            res = minimize(_objective, x0, method='Nelder-Mead',
                          options={'maxiter': maxiter, 'xatol': 1e-12, 'fatol': 1e-14,
                                   'adaptive': True})
        elif method == 'SLSQP':
            res = minimize(_objective, x0, method='SLSQP',
                          options={'maxiter': maxiter, 'ftol': 1e-14})
        else:
            res = minimize(_objective, x0, method='Nelder-Mead',
                          options={'maxiter': maxiter, 'xatol': 1e-12, 'fatol': 1e-14,
                                   'adaptive': True})
        return res.x, _score(res.x.reshape(_N, 2))
    except Exception:
        return x0, _score(x0.reshape(_N, 2))

def construct():
    """Return 8 distinct points maximizing the product of distances at fixed diameter."""
    n = _N
    rng = np.random.default_rng(12345)
    
    best_score = -1e10
    best_points = None
    
    # Phase 1: Multi-start local optimization with diverse initializations
    styles = [0, 1, 2, 3, 4, 5, 1, 5, 5, 5]
    for style in styles:
        if _remaining() < 15:
            break
        x0 = _make_cage_init(rng, style).flatten()
        # First refine with Nelder-Mead (good for non-smooth landscape)
        x1, s1 = _refine(x0, method='Nelder-Mead', maxiter=3000)
        # Then polish with L-BFGS-B
        x2, s2 = _refine(x1, method='L-BFGS-B', maxiter=5000)
        if s2 > best_score:
            best_score = s2
            best_points = x2.copy()
    
    # Phase 2: Differential Evolution for global search
    if _remaining() > 30:
        try:
            bounds = [(-10, 10)] * (2 * n)
            de_result = differential_evolution(
                _objective, bounds,
                seed=42, maxiter=200, tol=1e-10,
                popsize=15, mutation=(0.3, 0.8),
                recombination=0.7, polish=True,
                init='latinhypercube',
                updating='deferred'
            )
            s_de = _score(de_result.x.reshape(n, 2))
            if s_de > best_score:
                best_score = s_de
                best_points = de_result.x.copy()
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish
    if best_points is not None and _remaining() > 10:
        # SLSQP polish
        x3, s3 = _refine(best_points, method='SLSQP', maxiter=10000)
        if s3 > best_score:
            best_score = s3
            best_points = x3.copy()
        
        # Final L-BFGS-B with very tight tolerance
        x4, s4 = _refine(best_points, method='L-BFGS-B', maxiter=20000)
        if s4 > best_score:
            best_score = s4
            best_points = x4.copy()
    
    # Ensure we have a valid result
    if best_points is None:
        angles = 2 * np.pi * np.arange(n) / n
        best_points = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()
    
    pts = best_points.reshape(n, 2)
    
    # Ensure distinct points
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(pts[i] - pts[j]) < 1e-6:
                pts[j] += [1e-4, 0]
    
    # Ensure |coordinate| < 1e6
    pts = np.clip(pts, -1e5, 1e5)
    
    return pts.tolist()
# EVOLVE-BLOCK-END
