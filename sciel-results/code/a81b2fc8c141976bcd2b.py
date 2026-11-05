"""
Optimize 8 points to maximize the normalized product of pairwise distances
(discriminant / diameter^(n(n-1))).

Approach:
- Parameterize points in the plane (16 real coordinates).
- Objective: maximize log(product of pairwise distances) - n(n-1)*log(diameter).
- Use scipy.optimize with a soft-max diameter to keep it smooth.
- Multiple random restarts and local refinement.
"""

import numpy as np
from math import cos, sin, pi, log, sqrt
from scipy.optimize import minimize, differential_evolution

def _distances(points):
    """Return matrix of pairwise distances."""
    n = len(points)
    pts = np.array(points)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.linalg.norm(diff, axis=2)

def _raw_objective(x):
    """Raw objective for global search (DE). Minimize negative score."""
    n = 8
    pts = x.reshape(8, 2)
    dist = _distances(pts)
    np.fill_diagonal(dist, 0)
    dist = np.maximum(dist, 1e-15)
    
    iu = np.triu_indices(n, k=1)
    pair_dists = dist[iu]
    log_prod = np.sum(np.log(pair_dists))
    
    diam = np.max(dist)
    if diam < 1e-15:
        return 1e10
        
    val = log_prod - n * (n - 1) * np.log(diam)
    
    # Penalty for points being too close
    min_dist = np.min(pair_dists)
    if min_dist < 1e-6:
        val -= 1e6 * (1e-6 - min_dist)
        
    # Penalty for large coordinates (keep search bounded)
    if np.any(np.abs(pts) > 10):
        val -= 1e6 * (np.max(np.abs(pts)) - 10)

    return -val

def _smooth_objective(x):
    """Smooth objective for local refinement (L-BFGS-B)."""
    n = 8
    pts = x.reshape(8, 2)
    dist = _distances(pts)
    np.fill_diagonal(dist, 0)
    dist = np.maximum(dist, 1e-15)
    
    iu = np.triu_indices(n, k=1)
    pair_dists = dist[iu]
    log_prod = np.sum(np.log(pair_dists))
    
    # Soft max for diameter to maintain differentiability
    # Use a high beta to approximate max
    beta = 50.0
    # Shift for numerical stability
    max_d = np.max(dist)
    soft_diam = np.log(np.sum(np.exp(beta * (dist - max_d)))) / beta + max_d
    log_diam = np.log(soft_diam)
    
    val = log_prod - n * (n - 1) * log_diam
    
    min_dist = np.min(pair_dists)
    if min_dist < 1e-5:
        val -= 1e8 * (1e-5 - min_dist) ** 2

    return -val

def construct():
    """
    Returns 8 distinct points [x, y] that maximize the normalized product
    of pairwise distances.
    """
    n = 8
    best_score = -np.inf
    best_points = None
    
    # Reference: regular octagon
    reg_pts = np.array([[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)])
    dist_reg = _distances(reg_pts)
    np.fill_diagonal(dist_reg, 0)
    diam_reg = np.max(dist_reg)
    log_prod_reg = np.sum(np.log(np.maximum(dist_reg[np.triu_indices(n, k=1)], 1e-15)))
    ref_score = log_prod_reg - n * (n - 1) * np.log(diam_reg)
    best_score = ref_score
    best_points = reg_pts.copy()
    
    # 1. Global Search with Differential Evolution
    # Bounds: coordinates in [-2, 2] is sufficient as we can scale
    bounds = [(-2, 2)] * 16
    rng = np.random.default_rng(42)
    
    # DE is slow, so limit iterations
    # popsize=15, maxiter=100 should be fast enough
    try:
        de_result = differential_evolution(
            _raw_objective, 
            bounds=bounds, 
            seed=42, 
            maxiter=100, 
            popsize=15,
            tol=1e-4
        )
        
        if de_result.fun < -best_score:
            best_score = -de_result.fun
            best_points = de_result.x.reshape(8, 2)
    except Exception:
        pass

    # 2. Local Refinement with L-BFGS-B from multiple starts
    # Start from best DE result and Regular Octagon
    starts = [best_points.flatten(), reg_pts.flatten()]
    
    # Add a few random starts to ensure coverage
    for _ in range(5):
        angles = rng.uniform(0, 2 * pi, 8)
        radii = rng.uniform(0.5, 1.5, 8)
        starts.append(np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]).flatten())
        
    for x0 in starts:
        try:
            result = minimize(_smooth_objective, x0, method='L-BFGS-B', 
                             options={'maxiter': 2000, 'ftol': 1e-14})
            
            if result.fun < -best_score:
                best_score = -result.fun
                best_points = result.x.reshape(8, 2)
        except Exception:
            continue
            
    # Final check: ensure points are valid
    pts = best_points
    if np.any(np.abs(pts) >= 1e6):
        pts = reg_pts
    
    # Verify distinctness
    dist = _distances(pts)
    np.fill_diagonal(dist, 0)
    if np.min(dist) < 1e-8:
        pts = reg_pts
    
    return [[float(x), float(y)] for x, y in pts]
