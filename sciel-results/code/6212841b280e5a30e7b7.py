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
    T_start = 0.02
    T_end = 1e-8
    n_iterations = 150000
    step_size = 0.04
    
    # Cooling schedule
    alpha = (T_end / T_start) ** (1.0 / n_iterations)
    
    # Track recent improvements for adaptive step size
    no_improve_count = 0
    max_no_improve = 5000
    
    for it in range(n_iterations):
        T = T_start * (alpha ** it)
        
        # Pick a random center to perturb
        i = np.random.randint(N)
        old_pos = centers[i].copy()
        
        # Random perturbation with adaptive step size
        centers[i] += np.random.normal(0, step_size, 2)
        
        # Clip to stay within bounds
        centers[i] = np.clip(centers[i], -0.4999, 0.4999)
        
        new_score = compute_score(centers)
        
        if new_score >= current_score:
            current_score = new_score
            no_improve_count = 0
            if new_score > best_score:
                best_score = new_score
                best_centers = centers.copy()
        else:
            # Accept with probability exp((new_score - current_score) / T)
            delta = new_score - current_score
            if np.random.rand() < np.exp(delta / T):
                current_score = new_score
                no_improve_count = 0
            else:
                # Reject: revert
                centers[i] = old_pos
                no_improve_count += 1
        
        # Adaptive step size: increase if stuck, decrease if improving
        if no_improve_count > max_no_improve:
            step_size *= 1.2
            no_improve_count = 0
        elif it % 2000 == 0:
            step_size *= 0.98
    
    return best_centers
# EVOLVE-BLOCK-END
