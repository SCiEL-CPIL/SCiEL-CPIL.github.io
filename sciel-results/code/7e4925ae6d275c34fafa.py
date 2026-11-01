"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import random

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
    Fixes point 0 at (0,0,0), point 1 at (1,0,0), and point 2 at (0,1,0) to remove translational and most rotational symmetry.
    """
    points = np.zeros((N, D))
    points[0] = [0, 0, 0]
    points[1] = [1, 0, 0]
    points[2] = [0, 1, 0]
    points[3:] = flat.reshape(N-3, D)
    
    return _compute_score(points)

def _random_initialization():
    """Generate a random set of points in a reasonable range."""
    # Free points are 3 to N-1
    n_free = N - 3
    flat = np.random.uniform(-2, 2, size=(n_free, D))
    return flat

def construct():
    """
    Construct 14 points in 3D space to minimize the squared ratio of
    maximum to minimum pairwise distance.
    
    Uses Simulated Annealing for global search followed by L-BFGS for local polishing.
    """
    np.random.seed(42)
    random.seed(42)
    
    n_free = N - 3
    dim = n_free * D
    
    best_flat = _random_initialization()
    best_score = _objective_reduced(best_flat)
    
    # Simulated Annealing Parameters
    T = 5.0  # Initial temperature
    alpha = 0.98  # Cooling rate
    T_min = 0.001
    max_iter = 20000
    no_improve_limit = 500
    step_size = 0.5
    
    iter_count = 0
    no_improve_count = 0
    
    while T > T_min and iter_count < max_iter:
        # Generate neighbor
        # Adaptive step size: smaller when T is lower
        current_step = step_size * (T / 5.0 + 0.1)
        
        new_flat = best_flat + np.random.normal(0, current_step, size=dim)
        
        new_score = _objective_reduced(new_flat)
        
        delta = new_score - best_score
        
        # Acceptance criterion
        if delta < 0 or random.random() < np.exp(-delta / T):
            best_flat = new_flat
            best_score = new_score
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # Cooling
        T *= alpha
        
        # Restart logic if stuck
        if no_improve_count > no_improve_limit:
            # Perturb current best significantly or restart from scratch with higher T?
            # Let's perturb significantly to escape local minima
            best_flat = best_flat + np.random.normal(0, 1.0, size=dim)
            best_score = _objective_reduced(best_flat)
            T = max(T, 2.0)  # Reheat
            no_improve_count = 0
            
        iter_count += 1
        
    # Final Polishing with L-BFGS
    # Define bounds to keep points in a reasonable space
    bounds = [(-10, 10)] * dim
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
    pts[2] = [0, 1, 0]
    pts[3:] = best_flat.reshape(N-3, D)
    
    return pts
# EVOLVE-BLOCK-END
