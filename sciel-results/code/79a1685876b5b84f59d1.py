import numpy as np
import time

N = 200

def construct():
    # Initialize with a random uniform distribution
    # This is distinct from the grid-based start of the seed program's local search.
    # However, random start is very hard for 200 circles.
    # Let's use a coarse grid to ensure we start in a "reasonable" state, 
    # but the optimization is SA.
    
    # Let's use a 10x20 grid for init
    m = 10
    n_rows = m
    n_cols = N // m
    rem = N % m
    # Distribute
    centers = []
    dy = 1.0 / (n_rows + 1)
    dx = 1.0 / (n_cols + 1)
    
    # Simple grid
    for r in range(n_rows):
        y = -0.5 + dy * (r + 1)
        for c in range(n_cols):
            x = -0.5 + dx * (c + 1)
            centers.append([x, y])
            
    # Add remaining
    for r in range(rem):
        y = -0.5 + dy * (r + 1)
        x = -0.5 + dx * 0.5
        centers.append([x, y])
        
    P = np.array(centers, dtype=float)
    
    def get_score(P):
        # Vectorized min distance calculation
        # Shape (N, N, 2)
        diffs = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=-1))
        
        # Get upper triangle indices
        idx = np.triu_indices(N, 1)
        min_pair = np.min(dists[idx])
        
        # Boundary distances
        # Distance to each wall
        d_left = 0.5 - P[:, 0]
        d_right = 0.5 + P[:, 0]
        d_bottom = 0.5 - P[:, 1]
        d_top = 0.5 + P[:, 1]
        
        # Min distance to any wall for each point
        min_bound_per_point = np.min([d_left, d_right, d_bottom, d_top], axis=0)
        min_bound = np.min(min_bound_per_point)
        
        return min(min_pair, min_bound)

    current_score = get_score(P)
    
    # Simulated Annealing Parameters
    T = 0.005
    T_min = 1e-6
    cooling_rate = 0.995
    
    start_time = time.time()
    max_time = 110.0
    
    # Step size adaptation
    step_size = 0.05
    no_improve_count = 0
    best_score = current_score
    best_P = P.copy()
    
    while time.time() - start_time < max_time:
        # Pick random point
        i = np.random.randint(N)
        
        # Generate move
        # Use a larger step initially?
        move = np.random.randn(2) * step_size
        P_new = P.copy()
        P_new[i] += move
        
        # Optional: Keep within bounds? 
        # If we go out, score drops. SA will handle it.
        
        new_score = get_score(P_new)
        
        # Acceptance
        delta = new_score - current_score
        if delta > 0:
            P = P_new
            current_score = new_score
            no_improve_count = 0
            if current_score > best_score:
                best_score = current_score
                best_P = P.copy()
        else:
            # Probabilistic acceptance
            if T > 0:
                prob = np.exp(delta / T)
                if np.random.rand() < prob:
                    P = P_new
                    current_score = new_score
            no_improve_count += 1
            
        # Cooling
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
        # Step size adaptation
        # If stuck, increase step? Or decrease?
        # Usually, if stuck in local min, increase step to jump out.
        if no_improve_count > 500:
            step_size *= 1.1
            no_improve_count = 0
        elif no_improve_count > 100:
            step_size *= 0.9
            
        # Prevent step from becoming too large or small
        step_size = np.clip(step_size, 0.001, 0.1)

    return best_P
