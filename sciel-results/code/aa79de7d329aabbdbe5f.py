"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_radius(centers):
    """Compute the common radius for given centers."""
    # Distance between all pairs
    n = len(centers)
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    # Exclude self-distances (diagonal)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to walls: for each center and each axis, compute distance to nearest wall
    # Walls are at -0.5 and 0.5
    wall_dists = np.minimum(0.5 - centers, centers + 0.5)
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist, min_wall_dist) / 2.0

def _local_optimize(centers, max_time=100.0):
    """Local optimization to maximize the minimum radius."""
    start_time = time.time()
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    n = len(centers)
    # Simple iterative optimization: try small random perturbations
    # and accept if they improve the radius
    
    step_size = 0.05
    np.random.seed(42)
    
    while time.time() - start_time < max_time:
        improved = False
        for _ in range(50):  # Try 50 random directions per iteration
            if time.time() - start_time > max_time - 2:
                break
            
            # Random perturbation
            delta = np.random.randn(n, 3) * step_size
            new_centers = best_centers + delta
            
            # Clamp to valid range (with small margin)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            new_radius = _compute_radius(new_centers)
            
            if new_radius > best_radius + 1e-15:
                best_centers = new_centers
                best_radius = new_radius
                improved = True
                # Decrease step size on improvement
                step_size *= 0.95
        
        if not improved:
            # Reduce step size when stuck
            step_size *= 0.8
            if step_size < 1e-6:
                break
    
    return best_centers, best_radius

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    # Start with simple cubic grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    # Local optimization
    centers, _ = _local_optimize(centers, max_time=min(100.0, deadline - (time.time() - start_time)))
    
    return centers
# EVOLVE-BLOCK-END
