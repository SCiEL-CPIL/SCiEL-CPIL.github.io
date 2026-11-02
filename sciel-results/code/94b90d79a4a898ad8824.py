"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import itertools

def compute_score(pts):
    """
    Computes the squared ratio of maximum to minimum pairwise distance.
    Returns infinity if points are not distinct or invalid.
    """
    if len(pts) != N:
        return np.inf
    # Calculate all pairwise squared distances
    # pts shape (N, D)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    
    # Zero out diagonal (self-distances)
    np.fill_diagonal(dist_sq, np.inf)
    
    min_d_sq = np.min(dist_sq)
    max_d_sq = np.max(dist_sq)
    
    # Check for distinctness
    if min_d_sq < 1e-10:
        return np.inf
        
    if min_d_sq == 0 or np.isnan(min_d_sq) or np.isnan(max_d_sq):
        return np.inf
        
    return (max_d_sq / min_d_sq) ** 2

def objective_flat(x):
    """
    Objective function for the optimizer.
    Minimizes log(max_d_sq / min_d_sq) + penalty.
    """
    pts = x.reshape((N, D))
    
    # Calculate squared distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    
    min_d_sq = np.min(dist_sq)
    max_d_sq = np.max(dist_sq)
    
    # Penalty for points being too close
    # We want min_d_sq > epsilon
    # If min_d_sq is small, penalty is large
    eps = 1e-4
    if min_d_sq < eps:
        penalty = (eps - min_d_sq) ** 2 * 1e6
    else:
        penalty = 0.0
        
    # Objective: log(max/min)
    # If min is 0, log is -inf, but we have penalty.
    # To be safe, clip min_d_sq to a very small positive number for the log calculation
    safe_min = max(min_d_sq, 1e-12)
    
    val = np.log(max_d_sq / safe_min)
    
    return val + penalty

def construct():
    start_time = time.time()
    timeout = 145.0 # Safety margin for 150s limit
    
    # Initial guess: Grid-like or Halton
    # Let's try a simple grid first to ensure distinctness
    side = int(np.ceil(N ** (1 / D)))
    grid_coords = list(itertools.product(range(side), repeat=D))[:N]
    x0 = np.array(grid_coords, dtype=float)
    
    # Normalize x0 to be within [1, 2] to avoid boundary issues and scaling issues
    x0 = x0 * 0.5 + 1.0
    
    best_x = x0
    best_score = compute_score(x0)
    
    # Define bounds: [0.1, 5.0] for each coordinate
    bounds = [(0.1, 5.0)] * (N * D)
    
    # Optimization Loop
    # We will try to optimize using L-BFGS-B
    # Since the problem is non-convex, we might do a few restarts if time permits
    
    attempts = 0
    max_attempts = 5 # Don't do too many heavy restarts
    
    for i in range(max_attempts):
        if time.time() - start_time > timeout - 20:
            break
            
        # Perturb initial guess slightly for restarts
        if i == 0:
            x_start = x0.copy()
        else:
            # Random restart within bounds
            x_start = np.random.uniform(0.1, 5.0, size=(N, D))
        
        try:
            res = minimize(
                objective_flat,
                x_start.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.success or res.status in [0, 1, 2, 3, 4, 5]: # Common success statuses
                x_new = res.x.reshape((N, D))
                score = compute_score(x_new)
                
                if score < best_score:
                    best_score = score
                    best_x = x_new
                    
        except Exception:
            pass
        
        # If we found a very good score, we might stop early or refine
        if best_score < 2.0: # Heuristic: good ratio
             break
             
    # Final Polish: Run one more optimization with the best found point
    # using tighter tolerances if time allows
    if time.time() - start_time < timeout - 5:
        try:
            res_final = minimize(
                objective_flat,
                best_x.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            x_final = res_final.x.reshape((N, D))
            score_final = compute_score(x_final)
            if score_final < best_score:
                best_x = x_final
                best_score = score_final
        except Exception:
            pass
            
    return best_x
# EVOLVE-BLOCK-END
