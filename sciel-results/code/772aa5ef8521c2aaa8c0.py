"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_radius(centers):
    """Compute the maximum valid radius for given centers."""
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to walls
    wall_dists = np.min(np.abs(0.5 - np.abs(centers)), axis=1)
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist / 2.0, min_wall_dist)


def _force_directed_relax(centers, n_iters=500, learning_rate=0.01, decay=0.999):
    """Force-directed relaxation to maximize minimum distance."""
    centers = centers.copy()
    n = len(centers)
    lr = learning_rate
    
    for i in range(n_iters):
        # Compute forces from pairwise repulsion
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, 1.0)  # Avoid division by zero
        
        # Repulsive force: 1/d^2 direction
        forces = np.zeros_like(centers)
        for j in range(n):
            for k in range(j):
                d = dists[j, k]
                if d < 0.1:  # Only apply force for close pairs
                    force_mag = 1.0 / (d ** 2)
                    forces[j] += force_mag * (centers[j] - centers[k]) / d
                    forces[k] -= force_mag * (centers[j] - centers[k]) / d
        
        # Wall repulsion
        for j in range(n):
            for axis in range(3):
                dist_to_wall = 0.5 - abs(centers[j, axis])
                if dist_to_wall < 0.05:
                    force_mag = 1.0 / (dist_to_wall ** 2)
                    sign = 1.0 if centers[j, axis] >= 0 else -1.0
                    forces[j, axis] += sign * force_mag
        
        # Update positions
        centers += lr * forces
        
        # Clamp to valid region
        centers = np.clip(centers, -0.4999, 0.4999)
        
        # Decay learning rate
        lr *= decay
    
    return centers


def _local_refine(centers):
    """Local refinement using L-BFGS-B to maximize minimum radius."""
    n = len(centers)
    
    def objective(x):
        c = x.reshape(n, 3)
        # Negative radius (we minimize)
        r = _compute_radius(c)
        return -r
    
    # Initial bounds: centers must stay in [-0.5, 0.5]^3
    bounds = [(-0.5, 0.5)] * (n * 3)
    
    result = minimize(objective, centers.flatten(), method='L-BFGS-B', 
                      bounds=bounds, options={'maxiter': 100, 'ftol': 1e-12})
    
    return result.x.reshape(n, 3)


def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    # Start with grid initialization
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    best_radius = _compute_radius(centers)
    best_centers = centers.copy()
    
    # Phase 1: Force-directed relaxation
    elapsed = time.time() - start_time
    if elapsed < deadline - 60:
        centers = _force_directed_relax(centers, n_iters=300, learning_rate=0.005, decay=0.998)
        r = _compute_radius(centers)
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
    
    # Phase 2: Local refinement with L-BFGS-B
    elapsed = time.time() - start_time
    if elapsed < deadline - 30:
        centers = _local_refine(best_centers)
        r = _compute_radius(centers)
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
    
    # Phase 3: Final polish if time allows
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        centers = _force_directed_relax(best_centers, n_iters=100, learning_rate=0.001, decay=0.995)
        r = _compute_radius(centers)
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
        
        # Another L-BFGS pass
        elapsed = time.time() - start_time
        if elapsed < deadline - 5:
            centers = _local_refine(best_centers)
            r = _compute_radius(centers)
            if r > best_radius:
                best_radius = r
                best_centers = centers.copy()
    
    return best_centers
# EVOLVE-BLOCK-END
