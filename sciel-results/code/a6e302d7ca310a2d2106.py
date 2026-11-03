"""Packing 50 equal circles in unit square [-0.5, 0.5]^2 using simulated annealing."""
import numpy as np
import time

N = 50

def _compute_radius(centers):
    """Compute the largest valid radius for given centers."""
    # Min distance between pairs / 2
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    # Min distance to walls
    wall = 0.5 - np.max(np.abs(centers), axis=1)
    min_wall = np.min(wall)
    return min(min_pair, min_wall)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin
    
    # Initialize with a jittered grid to break symmetry
    m = int(np.ceil(np.sqrt(N)))  # 8
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    # Add small random perturbation to break grid symmetry
    rng = np.random.default_rng(42)
    centers += rng.normal(0, 0.005, size=centers.shape)
    
    # Clip to valid region
    centers = np.clip(centers, -0.5, 0.5)
    
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Simulated annealing parameters
    T = 0.05  # initial temperature
    T_min = 0.0001
    cooling = 0.9995
    step_size = 0.02
    
    # Precompute indices for pair distances
    n = N
    idx_i, idx_j = np.triu_indices(n, k=1)
    
    while time.time() < deadline:
        # Random perturbation
        delta = rng.normal(0, step_size, size=centers.shape)
        new_centers = centers + delta
        new_centers = np.clip(new_centers, -0.5, 0.5)
        
        # Compute new radius
        new_radius = _compute_radius(new_centers)
        
        # Accept if better or with probability exp((new - old) / T)
        if new_radius >= best_radius:
            centers = new_centers
            best_centers = new_centers.copy()
            best_radius = new_radius
        else:
            # Metropolis acceptance
            delta_r = new_radius - best_radius
            if delta_r < 0:
                prob = np.exp(delta_r / T) if T > 1e-10 else 0.0
                if rng.random() < prob:
                    centers = new_centers
                else:
                    # Keep old centers but maybe still update current
                    pass
            else:
                centers = new_centers
        
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
        
        # Occasionally reset temperature for exploration
        if rng.random() < 0.001:
            T = max(T, 0.01)
    
    return best_centers
