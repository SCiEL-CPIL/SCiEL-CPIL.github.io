import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initial positions
    # Random uniform in [-0.5, 0.5]
    # To avoid coincident centers, ensure small separation? 
    # SA will handle overlaps by rejecting or moving.
    # But if start with overlaps, r=0.
    # Better to start with a grid or random with min distance check?
    # Simple random is usually fine if we clamp and SA handles it.
    
    # Let's use a slightly better init: 
    # Place points on a grid, then add small random noise.
    # Or just random.
    
    pos = np.random.uniform(-0.5, 0.5, (N, 3))
    
    # Initial radius calculation
    def calc_radius(p):
        # Pairwise distances
        # Using broadcasting
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Get upper triangle
        i, j = np.triu_indices(N, 1)
        min_pair_dist = np.min(dists[i, j])
        
        # Boundary distances
        # Distance to each of the 6 faces
        # x: 0.5 - p[:,0], p[:,0] + 0.5
        # y: 0.5 - p[:,1], p[:,1] + 0.5
        # z: 0.5 - p[:,2], p[:,2] + 0.5
        b_x = np.min(np.vstack([0.5 - p[:, 0], p[:, 0] + 0.5]), axis=0)
        b_y = np.min(np.vstack([0.5 - p[:, 1], p[:, 1] + 0.5]), axis=0)
        b_z = np.min(np.vstack([0.5 - p[:, 2], p[:, 2] + 0.5]), axis=0)
        min_b = np.min(np.vstack([b_x, b_y, b_z]))
        
        return min(min_pair_dist / 2.0, min_b)
    
    best_pos = pos.copy()
    best_r = calc_radius(best_pos)
    
    # Simulated Annealing parameters
    T = 0.01  # Initial temperature
    T_min = 1e-6
    alpha = 0.999  # Cooling rate
    
    # Step size
    step_size = 0.05
    
    while time.time() < deadline:
        # Pick a random index
        idx = np.random.randint(N)
        
        # Generate perturbation
        # Gaussian noise
        noise = np.random.normal(0, step_size, 3)
        
        new_pos = pos.copy()
        new_pos[idx] += noise
        
        # Clamp to cube
        new_pos[idx] = np.clip(new_pos[idx], -0.5, 0.5)
        
        # Calculate new radius
        new_r = calc_radius(new_pos)
        
        # Acceptance criterion
        delta = new_r - best_r
        if delta >= 0 or np.random.rand() < np.exp(delta / T):
            pos = new_pos
            if new_r > best_r:
                best_r = new_r
                best_pos = pos.copy()
        
        # Cool down
        T *= alpha
        if T < T_min:
            # Restart or reheat? 
            # Reheat slightly to escape local minima
            T = T_min * 10
            # Also perturb best_pos slightly?
            # Or just continue with current pos (which is stuck)
            # Better: reset pos to best_pos with some noise
            pos = best_pos + np.random.normal(0, 0.1, (N, 3))
            pos = np.clip(pos, -0.5, 0.5)
            
    return best_pos
