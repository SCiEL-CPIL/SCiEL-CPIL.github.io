"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_score(centers):
    """Compute the minimum radius for the given centers."""
    # Check for non-finite values
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check bounds
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    # Compute pairwise distances
    from scipy.spatial.distance import pdist
    if N < 2:
        min_dist = np.inf
    else:
        dists = pdist(centers, metric='euclidean')
        min_dist = np.min(dists)
    # Distance to walls
    wall_dists = np.min(np.minimum(0.5 - np.abs(centers), 0.5 - np.abs(centers)), axis=1)
    min_wall = np.min(wall_dists)
    return min(min_dist / 2.0, min_wall)


def _repulsion_energy(centers, r_target=0.135):
    """Compute repulsion energy to push spheres apart."""
    from scipy.spatial.distance import pdist
    if N < 2:
        pair_energy = 0.0
    else:
        dists = pdist(centers, metric='euclidean')
        # Penalty for overlapping spheres
        overlaps = np.maximum(0, 2 * r_target - dists)
        pair_energy = np.sum(overlaps ** 2)
    # Penalty for being outside or too close to walls
    wall_violations = np.maximum(0, np.abs(centers) - (0.5 - r_target))
    wall_energy = np.sum(wall_violations ** 2)
    return pair_energy + 10 * wall_energy


def _optimize_from_initial(centers, time_budget=100):
    """Optimize centers using simulated annealing."""
    start_time = time.time()
    current = centers.copy()
    best = current.copy()
    best_score = _compute_score(current)
    
    # Simulated annealing parameters
    T = 0.01
    T_min = 1e-8
    cooling = 0.9999
    step_size = 0.01
    
    while T > T_min:
        if time.time() - start_time > time_budget:
            break
        
        # Propose a move
        idx = np.random.randint(N)
        delta = np.random.randn(3) * step_size
        new_centers = current.copy()
        new_centers[idx] += delta
        
        # Keep within bounds
        new_centers = np.clip(new_centers, -0.5 + 1e-6, 0.5 - 1e-6)
        
        # Accept if better or with probability exp(-deltaE/T)
        current_score = _compute_score(current)
        new_score = _compute_score(new_centers)
        delta_score = current_score - new_score
        
        if delta_score < 0 or np.random.random() < np.exp(-delta_score / T):
            current = new_centers
            if new_score > best_score:
                best_score = new_score
                best = new_centers.copy()
        
        T *= cooling
    
    return best, best_score


def construct():
    """Construct optimal sphere packing."""
    start_time = time.time()
    
    # Initialize with a BCC-like structure for better starting point
    m = 4  # 4x4x4 = 64 positions, use 50
    spacing = 1.0 / m
    centers = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                x = -0.5 + (i + 0.5) * spacing
                y = -0.5 + (j + 0.5) * spacing
                z = -0.5 + (k + 0.5) * spacing
                centers.append([x, y, z])
    
    centers = np.array(centers[:N])
    
    # Optimize
    best, best_score = _optimize_from_initial(centers, time_budget=110)
    
    return best
# EVOLVE-BLOCK-END
