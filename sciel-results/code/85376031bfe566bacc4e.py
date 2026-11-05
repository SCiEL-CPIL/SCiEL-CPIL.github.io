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

def _objective(x):
    """Objective to minimize: -log(prod dist) + n(n-1)*log(soft_diam).
    x is 16-vector: [x0,y0,x1,y1,...,x7,y7]
    """
    n = 8
    pts = x.reshape(8, 2)
    # Ensure points are distinct (add tiny penalty if too close)
    dist = _distances(pts)
    np.fill_diagonal(dist, 0)
    
    # Log of product of all pairwise distances
    # Product over ordered pairs = (product over unordered pairs)^2
    # So log(prod ordered) = 2 * log(prod unordered)
    # We maximize log(prod), so minimize -log(prod) + penalty
    
    # Handle zero distances
    dist = np.maximum(dist, 1e-15)
    
    # Product over unordered pairs
    iu = np.triu_indices(n, k=1)
    pair_dists = dist[iu]
    log_prod = np.sum(np.log(pair_dists))
    
    # Diameter (max distance)
    diam = np.max(dist)
    
    # Soft-max for smoothness: use a high temperature soft-max
    # Actually, just use the max directly; the non-smoothness is handled by the optimizer
    # But to be safe, use a soft-max with large beta
    beta = 10.0
    log_max_dist = np.log(np.sum(np.exp(beta * dist)) / n) / beta
    # Actually that's not quite right. Let's just use the actual max.
    # The objective is: log_prod - n(n-1)*log(diam)
    # We want to maximize this, so minimize its negative.
    
    val = log_prod - n * (n - 1) * np.log(diam)
    
    # Penalty for points being too close
    min_dist = np.min(pair_dists)
    if min_dist < 1e-6:
        val -= 1e6 * (1e-6 - min_dist)
    
    # Penalty for points going out of bounds
    if np.any(np.abs(pts) >= 1e6):
        val -= 1e10
    
    return -val

def construct():
    """
    Returns 8 distinct points [x, y] that maximize the normalized product
    of pairwise distances.
    
    Uses numerical optimization with multiple restarts.
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
    
    # Try multiple random initializations
    rng = np.random.default_rng(42)
    
    for restart in range(20):
        if restart == 0:
            # Start from regular octagon with small perturbation
            x0 = reg_pts.flatten() + rng.normal(0, 0.01, 16)
        else:
            # Random initialization on unit disk
            angles = rng.uniform(0, 2 * pi, 8)
            radii = rng.uniform(0.3, 1.0, 8)
            x0 = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]).flatten()
        
        # Use L-BFGS-B for local optimization
        result = minimize(_objective, x0, method='L-BFGS-B', 
                         options={'maxiter': 500, 'ftol': 1e-12})
        
        if result.success or result.fun < -best_score:
            score = -result.fun
            if score > best_score:
                best_score = score
                best_points = result.x.reshape(8, 2)
    
    # Try a few more refinements with tighter tolerances
    for _ in range(5):
        x0 = best_points.flatten()
        result = minimize(_objective, x0, method='L-BFGS-B',
                         options={'maxiter': 1000, 'ftol': 1e-14})
        if result.success and -result.fun > best_score:
            best_score = -result.fun
            best_points = result.x.reshape(8, 2)
    
    # Final check: ensure points are valid
    pts = best_points
    if np.any(np.abs(pts) >= 1e6):
        pts = reg_pts
    
    # Verify distinctness
    dist = _distances(pts)
    np.fill_diagonal(dist, 0)
    if np.min(dist) < 1e-8:
        pts = reg_pts
    
    # Scale to have diameter 2 for output (the scoring handles scaling,
    # but let's return points with reasonable coordinates)
    # Actually, the scoring scales the set to diameter 2, so we can return
    # points at any scale. Let's return them as-is.
    
    return [[float(x), float(y)] for x, y in pts]
