"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def _compute_radius(centers):
    """Compute the exact maximum common radius for given centers."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim == 1:
        centers = centers.reshape(-1, 2)
    
    if centers.shape[0] != N:
        return 0.0

    # Distance to walls
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    
    # Min distance to wall for each circle
    wall_dists = np.minimum(wall_dists_x, wall_dists_y)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    if N > 1:
        dists = cdist(centers, centers)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
    else:
        min_pair = np.inf
    
    return min(min_wall, min_pair)

def _smooth_objective(x):
    """
    Smooth surrogate for the radius maximization problem.
    We want to maximize r = min(wall_dists, pair_dists/2).
    We approximate the min with a soft-min (log-sum-exp trick) to make it differentiable.
    """
    centers = x.reshape(N, 2)
    
    # Wall distances
    wx = 0.5 - np.abs(centers[:, 0])
    wy = 0.5 - np.abs(centers[:, 1])
    wall_dists = np.minimum(wx, wy)
    
    # Pairwise distances
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    
    # Combine all constraints into one array
    # We want to maximize the minimum of these values.
    # softmin(A) = -logsumexp(-A)
    # We use a temperature parameter for the soft-min. Lower T -> sharper min.
    T = 0.01
    all_dists = np.concatenate([wall_dists, pair_dists.flatten()])
    
    # Avoid log(0) or negative values if points overlap significantly
    # If points overlap, dist is 0. We want the objective to be low.
    # Adding a small epsilon ensures numerical stability.
    all_dists_safe = np.maximum(all_dists, 1e-6)
    
    # Soft min
    log_sum_exp = np.log(np.sum(np.exp(-all_dists_safe / T)))
    soft_min = -T * log_sum_exp
    
    # We want to maximize soft_min, so minimize -soft_min
    return -soft_min

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin for 120s limit
    
    np.random.seed(42)
    
    # Initialize with a grid-like structure
    # 7x7 grid = 49 circles. Add 1 more.
    dx = 0.1428
    xs = np.array([-3, -2, -1, 0, 1, 2, 3]) * dx
    ys = np.array([-3, -2, -1, 0, 1, 2, 3]) * dx
    xx, yy = np.meshgrid(xs, ys)
    centers = np.column_stack((xx.ravel(), yy.ravel()))
    
    # Add 50th point
    # Place it in a hole or random nearby
    extra = np.random.uniform(-0.4, 0.4, size=(1, 2))
    centers = np.vstack([centers, extra])
    
    # Clamp to valid region
    centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
    centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)
    
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Use local optimization (L-BFGS-B) to polish the grid
    # We run multiple restarts with different initial perturbations to escape local minima
    for i in range(10):
        if time.time() > deadline:
            break
            
        # Create a perturbed initial guess
        if i == 0:
            x0 = best_centers.flatten()
        else:
            # Perturb the best solution found so far
            x0 = (best_centers + np.random.normal(0, 0.02, size=(N, 2))).flatten()
            # Clamp
            x0 = np.clip(x0, -0.5, 0.5)
        
        try:
            # Minimize the negative smooth objective
            res = minimize(_smooth_objective, x0, method='L-BFGS-B', 
                           options={'maxiter': 100, 'ftol': 1e-10})
            
            if res.success or res.fun < _smooth_objective(x0):
                new_centers = res.x.reshape(N, 2)
                # Clamp to ensure validity
                new_centers[:, 0] = np.clip(new_centers[:, 0], -0.5, 0.5)
                new_centers[:, 1] = np.clip(new_centers[:, 1], -0.5, 0.5)
                
                new_radius = _compute_radius(new_centers)
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_centers = new_centers.copy()
                    
        except Exception:
            pass
            
    return best_centers
# EVOLVE-BLOCK-END
