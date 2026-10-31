import numpy as np
import random

N = 21

def construct():
    best_score = -1
    best_result = None
    
    # Aspect ratios to try
    # W + H = 2. 
    aspect_ratios = [1.0, 0.9, 1.1, 0.85, 1.15]
    
    for w_ratio in aspect_ratios:
        W = w_ratio
        H = 2.0 - W
        
        # Vectorized score calculation
        def calc_score_vec(x, y):
            # Compute pairwise distances
            dx = x[:, np.newaxis] - x[np.newaxis, :]
            dy = y[:, np.newaxis] - y[np.newaxis, :]
            dists = np.sqrt(dx**2 + dy**2)
            np.fill_diagonal(dists, np.inf)
            
            # Min distance to any other circle for each i
            min_dists = np.min(dists, axis=1)
            
            # Wall distances
            wall_dists = np.minimum.reduce([x, W - x, y, H - y])
            
            # Radii
            r = np.minimum(wall_dists, min_dists / 2.0)
            
            return np.sum(r)
        
        # Run multiple restarts
        for restart in range(5):
            # Random Initialization
            x = np.random.uniform(0, W, N)
            y = np.random.uniform(0, H, N)
            
            current_x = x.copy()
            current_y = y.copy()
            current_score = calc_score_vec(current_x, current_y)
            
            best_x = current_x.copy()
            best_y = current_y.copy()
            best_local_score = current_score
            
            # Simulated Annealing Parameters
            T = 0.05
            cooling = 0.9995
            max_iters = 3000
            step_scale = 0.1 * min(W, H)
            
            for it in range(max_iters):
                # Pick random circle
                k = random.randint(0, N-1)
                
                # Perturb
                # Step size decreases slightly as T decreases
                step = step_scale * (0.5 + T)
                
                new_x = current_x.copy()
                new_y = current_y.copy()
                
                new_x[k] += random.uniform(-step, step)
                new_y[k] += random.uniform(-step, step)
                
                # Clamp to bounds
                new_x[k] = np.clip(new_x[k], 0, W)
                new_y[k] = np.clip(new_y[k], 0, H)
                
                new_score = calc_score_vec(new_x, new_y)
                
                delta = new_score - current_score
                
                # Metropolis acceptance
                if delta > 0 or random.random() < np.exp(delta / T):
                    current_x = new_x
                    current_y = new_y
                    current_score = new_score
                    
                    if current_score > best_local_score:
                        best_local_score = current_score
                        best_x = current_x.copy()
                        best_y = current_y.copy()
                
                T *= cooling
            
        if best_local_score > best_score:
            best_score = best_local_score
            # Calculate final radii for the best configuration
            dx = best_x[:, np.newaxis] - best_x[np.newaxis, :]
            dy = best_y[:, np.newaxis] - best_y[np.newaxis, :]
            dists = np.sqrt(dx**2 + dy**2)
            np.fill_diagonal(dists, np.inf)
            min_dists = np.min(dists, axis=1)
            wall_dists = np.minimum.reduce([best_x, W - best_x, best_y, H - best_y])
            r_final = np.minimum(wall_dists, min_dists / 2.0)
            
            best_result = np.column_stack((best_x, best_y, r_final))
            
    if best_result is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_result
