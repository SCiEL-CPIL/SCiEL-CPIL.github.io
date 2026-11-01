"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import scipy.optimize as opt

N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

def _soft_min(values, beta):
    """
    Compute the soft minimum of a set of values.
    values: array of shape (M,)
    beta: sharpness parameter
    """
    # For numerical stability, subtract the min before exp
    min_val = np.min(values)
    # If beta is very large, exp(-beta * (v - min)) can overflow or underflow.
    # Actually, we want min. 
    # softmin = -1/beta * log(sum(exp(-beta * v)))
    # To avoid overflow in exp(-beta * v), we can factor out exp(-beta * min_val).
    # sum(exp(-beta * v)) = exp(-beta * min_val) * sum(exp(-beta * (v - min_val)))
    # log(sum(...)) = -beta * min_val + log(sum(exp(-beta * (v - min_val))))
    # softmin = -1/beta * (-beta * min_val + log(...)) = min_val - 1/beta * log(sum(exp(-beta * (v - min_val))))
    
    shifted = -beta * (values - min_val)
    # shifted is <= 0. exp(shifted) is in (0, 1]. No overflow.
    # Underflow is fine (becomes 0).
    sum_exp = np.sum(np.exp(shifted))
    if sum_exp == 0:
        return min_val
    return min_val - (1.0 / beta) * np.log(sum_exp)

def _get_distances_and_bounds(centers):
    """
    centers: (N, 2)
    Returns:
      pair_dists_half: (N*(N-1)/2,)
      wall_dists: (N, 2)
    """
    n = len(centers)
    # Pairwise distances
    # Compute all pairwise differences
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Get upper triangle indices
    iu = np.triu_indices(n, k=1)
    pair_dists_half = dists[iu] / 2.0
    
    # Wall distances: min(0.5 - x, 0.5 + x) for each coord
    # centers are in [-0.5, 0.5]
    # dist to left wall: centers[:, 0] - (-0.5) = centers[:, 0] + 0.5
    # dist to right wall: 0.5 - centers[:, 0]
    # dist to bottom wall: centers[:, 1] + 0.5
    # dist to top wall: 0.5 - centers[:, 1]
    
    wall_dists = np.column_stack((
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1])
    ))
    
    return pair_dists_half, wall_dists

def _objective_flat(x, beta):
    """
    x: (2N,)
    Returns negative soft-min radius.
    """
    centers = x.reshape(N, 2)
    pair_dists_half, wall_dists = _get_distances_and_bounds(centers)
    
    # Combine all constraints into one vector
    # We want to maximize the minimum of all these values.
    # So we minimize the negative of the soft-min.
    all_vals = np.concatenate([pair_dists_half, wall_dists.ravel()])
    
    # If any value is negative (overlap or out of bounds), soft-min will be negative.
    # The optimizer will try to increase the minimum.
    sm = _soft_min(all_vals, beta)
    return -sm

def _compute_radius(centers):
    """
    Compute the true maximum radius for a given configuration.
    """
    pair_dists_half, wall_dists = _get_distances_and_bounds(centers)
    min_pair = np.min(pair_dists_half)
    min_wall = np.min(wall_dists)
    return min(min_pair, min_wall)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for 120s limit
    
    best_centers = None
    best_radius = -1.0
    
    # Initial guess: Grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    initial_grid = grid[:N]
    
    # Define bounds for optimizer
    # Centers must be in [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Optimization settings
    betas = [5.0, 20.0, 100.0, 500.0, 2000.0]
    
    # We will do several restarts
    # Each restart uses a slightly perturbed initial grid or random points
    # To save time, we use the previous best result as a starting point for the next restart's later beta stages?
    # No, independent restarts are better for exploration.
    
    restart_count = 0
    
    while time.time() < deadline:
        restart_count += 1
        
        # Generate initial configuration
        if restart_count == 1:
            x0 = initial_grid.flatten()
        else:
            # Jitter the grid or use random points
            # Random points can be bad for dense packing. Jittered grid is safer.
            # Let's try jittered grid with increasing jitter for later restarts?
            # Or just random points in a smaller box to encourage clustering?
            # Let's stick to jittered grid.
            jitter = 0.1 / (restart_count ** 0.5)
            x0 = initial_grid + np.random.normal(0, jitter, size=(N, 2))
            # Clamp to bounds
            x0 = np.clip(x0, -0.49, 0.49)
            x0 = x0.flatten()
        
        current_x = x0
        current_radius = _compute_radius(current_x.reshape(N, 2))
        
        if current_radius > best_radius:
            best_radius = current_radius
            best_centers = current_x.reshape(N, 2).copy()
        
        # Run optimization stages
        for beta in betas:
            if time.time() > deadline - 5.0:
                break
                
            try:
                res = opt.minimize(
                    _objective_flat, 
                    current_x, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                current_x = res.x
                current_radius = _compute_radius(current_x.reshape(N, 2))
                
                if current_radius > best_radius:
                    best_radius = current_radius
                    best_centers = current_x.reshape(N, 2).copy()
                    
            except Exception:
                # If optimization fails, break
                break
        
        # If we found a very good radius, we might break early?
        # The theoretical max is ~0.071377.
        if best_radius > 0.0713:
             break

    # Final polish on the best result found
    if best_centers is not None and time.time() < deadline - 2.0:
        x_polish = best_centers.flatten()
        # Use a very high beta for final refinement
        try:
            res = opt.minimize(
                _objective_flat, 
                x_polish, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            x_final = res.x
            r_final = _compute_radius(x_final.reshape(N, 2))
            if r_final > best_radius:
                best_centers = x_final.reshape(N, 2)
        except Exception:
            pass

    if best_centers is None:
        return initial_grid
        
    # Ensure output is valid (finite, within bounds)
    # Clip again just in case
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
# EVOLVE-BLOCK-END
