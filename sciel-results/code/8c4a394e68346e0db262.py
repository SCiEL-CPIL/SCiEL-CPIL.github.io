import numpy as np
import random
import time

N = 21

def construct():
    # We will try a few different aspect ratios and initializations
    # and keep the best result.
    best_score = -1
    best_result = None
    
    # Perimeter is 4, so 2*(W+H) = 4 => W+H = 2.
    # We can vary W from 0.8 to 1.2 (H = 2 - W)
    
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.2, 0.7, 1.3, 0.6, 1.4]
    
    # Time budget per aspect ratio
    time_budget_per_ratio = 10.0
    total_start_time = time.time()
    
    for w_ratio in aspect_ratios:
        W = w_ratio
        H = 2.0 - W
        
        # Check if we have time left
        if time.time() - total_start_time > 110.0:
            break
            
        # Initialize with force-directed layout
        # Start with random positions
        x = np.random.uniform(0, W, N)
        y = np.random.uniform(0, H, N)
        
        # Force-directed relaxation
        # Repulsion between circles, attraction to center
        for _ in range(1000):
            # Calculate forces
            fx = np.zeros(N)
            fy = np.zeros(N)
            
            # Pairwise repulsion
            # dx[i,j] = x[i] - x[j]
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist_sq = dx**2 + dy**2
            # Avoid division by zero
            dist_sq[dist_sq < 1e-6] = 1e-6
            dist = np.sqrt(dist_sq)
            
            # Force magnitude ~ 1/dist^2 (inverse square)
            # We want to keep them apart.
            force_mag = 1.0 / dist_sq
            # Set diagonal to 0
            np.fill_diagonal(force_mag, 0)
            
            fx += np.sum(force_mag * dx, axis=1)
            fy += np.sum(force_mag * dy, axis=1)
            
            # Attraction to center
            cx = W / 2.0
            cy = H / 2.0
            fx += -0.01 * (x - cx)
            fy += -0.01 * (y - cy)
            
            # Update positions
            x += fx * 0.05
            y += fy * 0.05
            
            # Clamp to bounds
            x = np.clip(x, 0, W)
            y = np.clip(y, 0, H)
        
        # Simulated Annealing
        def compute_radii(x, y, W, H):
            # Compute distances between all pairs
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist_sq = dx**2 + dy**2
            dist = np.sqrt(dist_sq)
            
            # Neighbor constraints: r_i <= dist(i,j)/2
            # For each i, r_i is min over j!=i of dist(i,j)/2
            # Also wall constraints: r_i <= min(x_i, W-x_i, y_i, H-y_i)
            
            # Initialize radii with wall constraints
            r = np.minimum(np.minimum(x, W - x), np.minimum(y, H - y))
            
            # Pairwise constraints
            # dist is NxN matrix. dist[i,j] is distance between i and j.
            # We want min_j dist[i,j] / 2 for j != i.
            # Set diagonal to inf so it doesn't affect min
            dist_for_min = dist.copy()
            np.fill_diagonal(dist_for_min, np.inf)
            
            min_dist = np.min(dist_for_min, axis=1)
            r = np.minimum(r, min_dist / 2.0)
            
            return r
        
        r = compute_radii(x, y, W, H)
        current_score = np.sum(r)
        
        # SA Parameters
        T = 0.1
        T_min = 1e-5
        cooling_rate = 0.995
        max_iters = 20000
        start_time = time.time()
        
        for it in range(max_iters):
            if time.time() - start_time > time_budget_per_ratio:
                break
                
            # Pick random circle
            k = np.random.randint(0, N)
            
            # Perturbation size depends on T
            step_size = T * 0.5
            
            old_xk = x[k]
            old_yk = y[k]
            
            # New position
            x_new = x.copy()
            y_new = y.copy()
            
            x_new[k] = old_xk + np.random.uniform(-step_size, step_size)
            y_new[k] = old_yk + np.random.uniform(-step_size, step_size)
            
            # Clamp
            x_new[k] = np.clip(x_new[k], 0, W)
            y_new[k] = np.clip(y_new[k], 0, H)
            
            # Calculate new score
            r_new = compute_radii(x_new, y_new, W, H)
            new_score = np.sum(r_new)
            
            delta = new_score - current_score
            
            if delta > 0 or np.random.rand() < np.exp(delta / T):
                # Accept
                x = x_new
                y = y_new
                r = r_new
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_result = np.column_stack((x, y, r))
            
            # Cool down
            T *= cooling_rate
            if T < T_min:
                break
                
    # Ensure perimeter constraint is met exactly or less
    # The radii are calculated based on W, H where W+H=2.
    # So perimeter is 4.
    
    if best_result is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_result
