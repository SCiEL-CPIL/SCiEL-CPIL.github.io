"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Use simulated annealing to maximize the minimum distance between circles
    and from walls for 50 equal circles in a unit square [-0.5, 0.5]^2.
    
    The score is r = min(min_{i<j} |c_i - c_j| / 2, min_i min_k (0.5 - |c_ik|)).
    We want to maximize r.
    
    Approach:
    1. Initialize with a hexagonal-like grid.
    2. Use simulated annealing with random perturbations.
    3. Objective: maximize the minimum of pairwise distances (divided by 2) and wall distances.
    4. Use a cooling schedule and accept worse moves with probability exp(delta/T).
    """
    np.random.seed(42)
    
    # Initial placement: hexagonal grid for better packing
    centers = np.zeros((N, 2))
    
    # Hexagonal packing initialization
    # Use spacing that fits ~50 circles in [-0.5, 0.5]^2
    spacing = 0.142
    row_spacing = spacing * np.sqrt(3) / 2
    
    idx = 0
    y = -0.47
    row = 0
    while idx < N and y <= 0.47:
        x_offset = 0.0 if row % 2 == 0 else spacing / 2
        x = -0.47 + x_offset
        while x <= 0.47 and idx < N:
            centers[idx] = [x, y]
            idx += 1
            x += spacing
        y += row_spacing
        row += 1
    
    # Fill remaining if any (shouldn't happen with proper spacing)
    if idx < N:
        # Place remaining in a grid
        remaining = N - idx
        side = int(np.ceil(np.sqrt(remaining)))
        ticks = (np.arange(side) + 0.5) / side - 0.5
        for i in range(remaining):
            centers[idx + i] = [ticks[i % side], ticks[i // side]]
    
    # Clip to stay within bounds
    centers = np.clip(centers, -0.4999, 0.4999)
    
    # Compute initial score
    def compute_score(centers):
        """Compute the radius r for the given centers."""
        n = len(centers)
        if n < 2:
            return 0.5
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        # Set diagonal to inf
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        # Wall distances: min over all centers and all 4 walls
        min_wall = np.min(0.5 - np.abs(centers))
        
        return min(min_pair, min_wall)
    
    current_score = compute_score(centers)
    best_centers = centers.copy()
    best_score = current_score
    
    # Simulated annealing parameters
    T_start = 0.015
    T_end = 1e-7
    n_iterations = 100000
    step_size = 0.035
    
    # Cooling schedule
    alpha = (T_end / T_start) ** (1.0 / n_iterations)
    
    for it in range(n_iterations):
        T = T_start * (alpha ** it)
        
        # Pick a random center to perturb
        i = np.random.randint(N)
        old_pos = centers[i].copy()
        
        # Random perturbation
        centers[i] += np.random.normal(0, step_size, 2)
        
        # Clip to stay within bounds
        centers[i] = np.clip(centers[i], -0.4999, 0.4999)
        
        new_score = compute_score(centers)
        
        if new_score >= current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_centers = centers.copy()
        else:
            # Accept with probability exp((new_score - current_score) / T)
            delta = new_score - current_score
            if np.random.rand() < np.exp(delta / T):
                current_score = new_score
            else:
                # Reject: revert
                centers[i] = old_pos
        
        # Adaptive step size: reduce gradually
        if it % 1000 == 0:
            step_size *= 0.96
    
    return best_centers
# EVOLVE-BLOCK-END
