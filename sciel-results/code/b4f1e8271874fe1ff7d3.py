"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Initial grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    # Precompute pair indices for distance calculation
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    def calc_radius(c):
        # Vectorized pairwise distances
        diffs = c[i_idx] - c[j_idx]
        dists = np.linalg.norm(diffs, axis=1)
        min_pair_dist = np.min(dists)
        
        # Wall distances
        # c is in [-0.5, 0.5]
        # Distance to wall is 0.5 - |c|
        wall_dists = 0.5 - np.abs(c)
        min_wall_dist = np.min(wall_dists)
        
        r = 0.5 * min(min_pair_dist, 2 * min_wall_dist)
        return r

    current_r = calc_radius(centers)
    best_centers = centers.copy()
    best_r = current_r
    
    # Simulated Annealing parameters
    T_start = 0.01
    T_end = 1e-6
    alpha = 0.995 # cooling rate per step
    T = T_start
    
    # Step size for moves
    step_size = 0.1
    
    while time.time() < deadline:
        # Choose a random sphere to move
        idx = np.random.randint(N)
        
        # Generate a move
        # With high probability, small move; with low probability, large jump
        if np.random.rand() < 0.1:
            # Large jump: random position in box
            new_pos = np.random.uniform(-0.5, 0.5, 3)
        else:
            # Small move
            move = np.random.normal(0, step_size, 3)
            new_pos = centers[idx] + move
            # Clamp to box (optional, but helps keep validity)
            new_pos = np.clip(new_pos, -0.5, 0.5)
            
        # Try the move
        old_pos = centers[idx].copy()
        centers[idx] = new_pos
        
        new_r = calc_radius(centers)
        
        # Acceptance criterion
        if new_r > current_r:
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_centers = centers.copy()
        else:
            # Probability of accepting worse move
            delta_r = current_r - new_r
            # Scale delta_r for temperature
            # If delta_r is small, accept more likely
            prob = np.exp(-delta_r / (T + 1e-9))
            if np.random.rand() < prob:
                current_r = new_r
            else:
                centers[idx] = old_pos
        
        # Cool down
        T *= alpha
        if T < T_end:
            T = T_end
            
        # Adapt step size
        # If we are stuck, increase step size? Or decrease if too many rejections?
        # Simple heuristic: if best_r hasn't improved in a while, increase T or step size
        # For now, keep it simple.
        
        # Occasionally, if stuck, try to "shake" the system
        # This is handled by the large jumps above.

    return best_centers
# EVOLVE-BLOCK-END
