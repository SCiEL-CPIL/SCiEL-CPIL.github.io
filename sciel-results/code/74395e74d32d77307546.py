"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import itertools

def _pairwise_distances_sq(centers):
    """Compute squared pairwise distances matrix."""
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    return np.sum(diff * diff, axis=2)

def _compute_radius(centers):
    """Compute the valid radius for given centers."""
    n = len(centers)
    # Check boundaries
    min_boundary = 0.5 - np.max(np.abs(centers), axis=1)
    r_boundary = np.min(min_boundary)
    
    if r_boundary <= 0:
        return 0.0
    
    # Check pairwise distances
    dist_sq = _pairwise_distances_sq(centers)
    np.fill_diagonal(dist_sq, np.inf)
    min_dist_sq = np.min(dist_sq)
    r_pair = np.sqrt(min_dist_sq) / 2.0
    
    return min(r_boundary, r_pair)

def _objective(params, n=N):
    """Objective function to minimize (negative radius)."""
    centers = params.reshape(n, 3)
    r = _compute_radius(centers)
    return -r

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    np.random.seed(42)
    
    # Start with a reasonable initial configuration
    # Use a 5x5x4 grid arrangement
    best_centers = None
    best_radius = 0.0
    
    # Generate multiple random initial configurations
    n_configs = 5
    configs = []
    
    for config_idx in range(n_configs):
        if config_idx == 0:
            # Grid-like initial configuration
            m = 5
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y, z) for z in ticks[:4] for y in ticks for x in ticks])
            centers = grid[:N].copy()
        else:
            # Random initial configuration with some structure
            centers = np.random.uniform(-0.45, 0.45, size=(N, 3))
        
        configs.append(centers)
    
    for centers in configs:
        if time.time() > deadline - 20:
            break
        
        # Run optimization
        try:
            result = minimize(
                _objective,
                centers.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            opt_centers = result.x.reshape(N, 3)
            opt_radius = _compute_radius(opt_centers)
            
            if opt_radius > best_radius:
                best_radius = opt_radius
                best_centers = opt_centers.copy()
                
        except Exception as e:
            continue
    
    # Final polish with more iterations if time permits
    if best_centers is not None and time.time() < deadline - 10:
        try:
            result = minimize(
                _objective,
                best_centers.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            opt_centers = result.x.reshape(N, 3)
            opt_radius = _compute_radius(opt_centers)
            
            if opt_radius > best_radius:
                best_radius = opt_radius
                best_centers = opt_centers.copy()
                
        except Exception as e:
            pass
    
    if best_centers is None:
        # Fallback to simple grid
        m = 5
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks[:4] for y in ticks for x in ticks])
        best_centers = grid[:N]
    
    # Ensure all centers are strictly inside the cube
    eps = 1e-9
    best_centers = np.clip(best_centers, -0.5 + eps, 0.5 - eps)
    
    return best_centers
# EVOLVE-BLOCK-END
