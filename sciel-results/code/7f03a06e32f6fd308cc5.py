import numpy as np
import time
from scipy.optimize import minimize

N = 100
CUBE_HALF = 0.5

def _compute_radius(centers):
    """Compute the common radius for a set of centers."""
    # Wall clearance: min over i,k of (0.5 - |c_ik|)
    wall_clear = 0.5 - np.max(np.abs(centers), axis=1)
    min_wall = np.min(wall_clear)
    
    # Inter-sphere clearance: min over i<j of |c_i - c_j| / 2
    if len(centers) > 1:
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
    else:
        min_pair = min_wall
    
    return min(min_pair, min_wall)

def _soft_min_objective(x, beta):
    """
    Objective function for L-BFGS-B.
    Returns the negative soft-min of the distances.
    Minimizing this maximizes the soft-min distance.
    """
    c = x.reshape(N, 3)
    
    # Pairwise distances / 2
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    # Avoid sqrt for speed if possible, but we need actual distances for the log-sum-exp
    # We can compute sqrt only for the upper triangle to save time? 
    # N=100 -> 100x100 matrix is 10k elements. Sqrt is fast.
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    
    # Wall distances
    abs_c = np.abs(c)
    max_coord = np.max(abs_c, axis=1)
    wall_vals = 0.5 - max_coord
    
    # Combine all distances
    # We want to minimize -soft_min(all_vals)
    # soft_min = -1/beta * log(sum(exp(-beta * vals)))
    
    all_vals = np.concatenate([pair_vals.ravel(), wall_vals.ravel()])
    all_vals = all_vals[np.isfinite(all_vals)]
    
    if len(all_vals) == 0:
        return 0.0
        
    # Numerical stability: shift by max
    max_val = np.max(all_vals)
    shifted = -beta * (all_vals - max_val)
    # Clip to avoid underflow in exp
    shifted = np.clip(shifted, -700, 0)
    
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_val = max_val - log_sum / beta
    
    return -soft_min_val

def _make_initial_guess(seed):
    """
    Generate an initial guess using a jittered grid.
    A 5x5x5 grid has 125 points. We take the first 100.
    """
    rng = np.random.RandomState(seed)
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    # Generate grid points
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    # Take first N points
    grid = grid[:N]
    # Add random jitter to break symmetry and help escape local minima
    # Jitter magnitude should be small enough to stay in valid region
    # but large enough to perturb the lattice
    jitter = rng.uniform(-0.01, 0.01, size=(N, 3))
    grid = grid + jitter
    # Ensure within bounds
    grid = np.clip(grid, -0.5, 0.5)
    return grid.flatten()

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    best_r = 0.0
    best_c = None
    
    # List of betas to try, increasing sharpness
    # Low beta: smooth, good for global search
    # High beta: sharp, good for local refinement
    betas = [50, 100, 200, 500, 1000]
    
    # Number of random seeds to try
    num_seeds = 10
    
    for seed in range(num_seeds):
        if time.time() - start_time > time_limit - 5:
            break
            
        x0 = _make_initial_guess(seed)
        x_current = x0.copy()
        
        for beta in betas:
            if time.time() - start_time > time_limit - 5:
                break
                
            # Define objective with current beta
            def obj(x):
                return _soft_min_objective(x, beta)
                
            # Use L-BFGS-B
            # maxiter and maxfev depend on beta and remaining time
            # For lower beta, we can afford more iterations
            max_iter = 200
            max_fev = 2000
            
            try:
                res = minimize(
                    obj,
                    x_current,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': max_iter, 'maxfev': max_fev, 'ftol': 1e-10, 'gtol': 1e-8}
                )
            except Exception:
                continue
                
            if res.success or res.fun < 1e-8:
                c_test = res.x.reshape(N, 3)
                r_test = _compute_radius(c_test)
                
                if r_test > best_r:
                    best_r = r_test
                    best_c = c_test.copy()
                    
                # Update current position for next beta
                # Only update if the new position is better or similar
                # Actually, always update to refine the local minimum
                x_current = res.x
            else:
                # If optimization failed, try to continue with next beta anyway?
                # Or break?
                # Let's break if it fails significantly
                if res.status not in [0, 1, 2, 3, 4, 5]: # 0: success, 1: max iter, etc
                     break
                else:
                     x_current = res.x

    # Final Polish:
    # Use the best configuration found and run a final optimization with a very high beta
    # or use a local search on the exact objective?
    # L-BFGS-B on the soft-min with high beta is effectively a local search on the min.
    
    if best_c is not None and time.time() - start_time < time_limit - 10:
        x_polish = best_c.flatten()
        # Try one more polish with very high beta
        beta_polish = 2000
        def obj_polish(x):
            return _soft_min_objective(x, beta_polish)
            
        try:
            res_polish = minimize(
                obj_polish,
                x_polish,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'maxfev': 5000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res_polish.success or res_polish.fun < 1e-10:
                c_polish = res_polish.x.reshape(N, 3)
                r_polish = _compute_radius(c_polish)
                if r_polish > best_r:
                    best_r = r_polish
                    best_c = c_polish.copy()
        except Exception:
            pass

    # Fallback if no valid solution was found
    if best_c is None:
        m = 5
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_c = grid[:N]

    return best_c
