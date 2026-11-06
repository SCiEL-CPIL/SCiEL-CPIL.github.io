"""Circle packing: 50 equal circles in unit square [-1/2, 1/2]^2."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the maximum radius for given centers."""
    # Distance between centers
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to walls
    wall_dists = np.min(np.stack([
        0.5 - centers[:, 0],
        0.5 + centers[:, 0],
        0.5 - centers[:, 1],
        0.5 + centers[:, 1]
    ], axis=1), axis=1)
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist, min_wall_dist) / 2.0

def _initial_grid():
    """Create initial grid arrangement."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].astype(float)

def _random_initializer(rng):
    """Create random initial positions."""
    return rng.uniform(-0.45, 0.45, size=(N, 2))

def _local_search(centers, max_iter=2000, lr=0.01, deadline=None):
    """Local search to improve circle packing."""
    centers = centers.copy()
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    
    for i in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute gradients numerically
        grad = np.zeros_like(centers)
        eps = 1e-5
        for idx in range(N):
            for dim in range(2):
                orig = centers[idx, dim]
                centers[idx, dim] = orig + eps
                r_plus = _compute_radius(centers)
                centers[idx, dim] = orig - eps
                r_minus = _compute_radius(centers)
                centers[idx, dim] = orig
                grad[idx, dim] = (r_plus - r_minus) / (2 * eps)
        
        # Gradient ascent step
        centers += lr * grad
        
        # Clamp to valid range
        centers = np.clip(centers, -0.499, 0.499)
        
        r = _compute_radius(centers)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
    
    return best_centers, best_r

def _evaluate_and_optimize(centers, deadline=None):
    """Run optimization from initial positions."""
    best_centers = centers.copy()
    best_r = _compute_radius(centers)
    
    # Try local search with different learning rates
    for lr in [0.005, 0.01, 0.02, 0.05]:
        if deadline is not None and time.time() > deadline:
            break
        result_centers, result_r = _local_search(centers, max_iter=500, lr=lr, deadline=deadline)
        if result_r > best_r:
            best_r = result_r
            best_centers = result_centers
    
    return best_centers, best_r

def construct():
    """Construct centers of 50 circles in unit square."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Start with grid
    grid_centers = _initial_grid()
    grid_r = _compute_radius(grid_centers)
    
    best_centers = grid_centers.copy()
    best_r = grid_r
    
    # Try multiple random restarts
    rng = np.random.default_rng(42)
    n_restarts = 10
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Random initial positions
        random_centers = _random_initializer(rng)
        
        # Optimize from this starting point
        opt_centers, opt_r = _evaluate_and_optimize(random_centers, deadline=deadline)
        
        if opt_r > best_r:
            best_r = opt_r
            best_centers = opt_centers
    
    # Final polish with smaller steps
    if time.time() < deadline - 10:
        final_centers, final_r = _local_search(best_centers, max_iter=1000, lr=0.001, deadline=deadline)
        if final_r > best_r:
            best_r = final_r
            best_centers = final_centers
    
    return best_centers
# EVOLVE-BLOCK-END
