"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import random

def _initial_grid(n):
    """Generate an initial grid of n points in [-0.5, 0.5]^3."""
    m = int(np.ceil(n ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n]

def _objective(params, n):
    """
    Objective function to minimize: negative of the minimum distance constraint.
    We maximize r = min(min_pairwise_dist/2, min_boundary_dist).
    Equivalently, minimize -r.
    """
    pts = params.reshape(n, 3)
    # Boundary distances: min over i,k of (0.5 - |c_ik|)
    # This is 0.5 - max over i,k of |c_ik|
    max_abs = np.max(np.abs(pts))
    boundary_r = 0.5 - max_abs
    
    # Pairwise distances
    # Compute all pairwise distances efficiently
    # diff[i,j,k] = pts[i,k] - pts[j,k]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    # Set diagonal to inf to ignore self-distance
    np.fill_diagonal(dists, np.inf)
    min_pairwise = np.min(dists)
    pairwise_r = min_pairwise / 2.0
    
    r = min(boundary_r, pairwise_r)
    if r <= 0:
        # Points are overlapping or outside; penalize heavily
        return -r + 100.0
    return -r

def construct():
    n = N
    # Start from a simple grid
    pts = _initial_grid(n).copy()
    
    # Use multiple random restarts with local optimization
    best_pts = pts.copy()
    best_r = None
    
    # Compute initial radius
    def _compute_r(pts):
        max_abs = np.max(np.abs(pts))
        boundary_r = 0.5 - max_abs
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pairwise = np.min(dists)
        pairwise_r = min_pairwise / 2.0
        return min(boundary_r, pairwise_r)
    
    best_r = _compute_r(best_pts)
    
    # Perform several rounds of optimization
    random.seed(42)
    np.random.seed(42)
    
    for restart in range(20):
        if restart == 0:
            current_pts = best_pts.copy()
        else:
            # Perturb the best solution found so far
            current_pts = best_pts + np.random.normal(0, 0.01, size=(n, 3))
            # Keep within bounds
            current_pts = np.clip(current_pts, -0.5, 0.5)
        
        # Flatten for optimizer
        x0 = current_pts.flatten()
        
        try:
            result = minimize(
                _objective,
                x0,
                args=(n,),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (n * 3),
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < best_r if best_r is not None else True:
                new_pts = result.x.reshape(n, 3)
                new_r = _compute_r(new_pts)
                if best_r is None or new_r > best_r:
                    best_r = new_r
                    best_pts = new_pts
        except Exception:
            continue
    
    # Final refinement with more iterations on the best solution
    x0 = best_pts.flatten()
    try:
        result = minimize(
            _objective,
            x0,
            args=(n,),
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (n * 3),
            options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        new_pts = result.x.reshape(n, 3)
        new_r = _compute_r(new_pts)
        if new_r > best_r:
            best_r = new_r
            best_pts = new_pts
    except Exception:
        pass
    
    # Ensure points are strictly inside the cube (with small margin for safety)
    eps = 1e-10
    best_pts = np.clip(best_pts, -0.5 + eps, 0.5 - eps)
    
    return best_pts
# EVOLVE-BLOCK-END
