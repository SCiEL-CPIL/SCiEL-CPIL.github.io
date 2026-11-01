"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize

def _compute_score(points):
    """Compute the squared ratio of max to min pairwise distance."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    iu = np.triu_indices(N, k=1)
    dist_sq_upper = dist_sq[iu]
    
    if len(dist_sq_upper) == 0:
        return 1e10
    
    min_d_sq = np.min(dist_sq_upper)
    max_d_sq = np.max(dist_sq_upper)
    
    if min_d_sq <= 1e-12:
        return 1e10
    
    return max_d_sq / min_d_sq

def _objective_reduced(flat):
    """
    Objective function for reduced parameterization.
    Fixes point 0 at (0,0,0) and point 1 at (1,0,0) to remove translational and rotational symmetry.
    """
    points = np.zeros((N, D))
    points[0] = [0, 0, 0]
    points[1] = [1, 0, 0]
    points[2:] = flat.reshape(N-2, D)
    
    return _compute_score(points)

def construct():
    """
    Construct 14 points in 3D space to minimize the squared ratio of
    maximum to minimum pairwise distance.
    
    Uses Differential Evolution for global search followed by L-BFGS-B for local polishing.
    """
    bounds = [(-3, 3)] * ((N-2) * D)
    
    # Step 1: Global Optimization with Differential Evolution
    # DE is stochastic, so we fix the seed for reproducibility
    # popsize=15 is relatively small for speed, maxiter=100
    # tol=1e-8
    res_de = differential_evolution(
        _objective_reduced, 
        bounds, 
        seed=42, 
        maxiter=100, 
        popsize=15, 
        tol=1e-8,
        polish=True, 
        init='sobol'
    )
    
    best_flat = res_de.x
    best_score = res_de.fun
    
    # Step 2: Local Polishing with L-BFGS-B
    # Use the DE result as the starting point for a high-precision local optimization
    res_lbfgs = minimize(
        _objective_reduced, 
        best_flat, 
        method='L-BFGS-B', 
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    if res_lbfgs.fun < best_score:
        best_flat = res_lbfgs.x
        best_score = res_lbfgs.fun
    
    # Reconstruct points
    pts = np.zeros((N, D))
    pts[0] = [0, 0, 0]
    pts[1] = [1, 0, 0]
    pts[2:] = best_flat.reshape(N-2, D)
    
    return pts
# EVOLVE-BLOCK-END
