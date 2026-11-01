import numpy as np
import time

N = 32

def sum_radii(c):
    """
    c: flat array of length 2*N, representing N centers (x, y)
    Returns: sum of inradii of the Voronoi cells clipped to [0,1]^2
    """
    p = c.reshape(N, 2)
    # Distances to walls
    dist_walls = np.minimum(p, 1 - p) # shape (N, 2)
    min_wall_dist = np.min(dist_walls, axis=1) # shape (N,)
    
    # Distances to other centers / 2
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_neighbor_dist = np.min(dists, axis=1) / 2.0
    
    radii = np.minimum(min_wall_dist, min_neighbor_dist)
    return np.sum(radii)

def construct():
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer
    
    # Initialize best with grid
    side = int(np.ceil(np.sqrt(N)))
    xs = (np.arange(side) + 0.5) / side
    ys = (np.arange(side) + 0.5) / side
    init_points = []
    for i in range(side):
        for j in range(side):
            if len(init_points) < N:
                init_points.append([xs[i], ys[j]])
    
    best_c = np.array(init_points).flatten()
    best_val = sum_radii(best_c)
    
    # Run multiple restarts with Simulated Annealing
    n_restarts = 15
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Initialization: Grid with noise
        c = best_c.copy() # Start from previous best or grid?
        # Let's start from grid + noise for diversity
        c = np.array(init_points).flatten()
        noise = np.random.normal(0, 0.05, size=c.shape)
        c = np.clip(c + noise, 0.01, 0.99)
        
        # Simulated Annealing
        T = 0.01
        T_min = 1e-6
        alpha = 0.995
        current_val = sum_radii(c)
        current_c = c
        
        max_iter = 20000
        for i in range(max_iter):
            if time.time() - start_time > time_limit:
                break
            if T < T_min:
                break
                
            # Perturb a random subset of centers
            n_perturb = np.random.randint(1, 4)
            indices = np.random.choice(N, n_perturb, replace=False)
            
            new_c = current_c.copy()
            for idx in indices:
                # Random step
                step = np.random.normal(0, 0.05, 2)
                new_c[2*idx:2*idx+2] += step
                # Clamp
                new_c[2*idx] = np.clip(new_c[2*idx], 0.01, 0.99)
                new_c[2*idx+1] = np.clip(new_c[2*idx+1], 0.01, 0.99)
            
            new_val = sum_radii(new_c)
            delta = new_val - current_val
            
            if delta > 0 or np.random.random() < np.exp(delta / T):
                current_c = new_c
                current_val = new_val
                
                if current_val > best_val:
                    best_val = current_val
                    best_c = current_c.copy()
            
            T *= alpha
        
        # Local Polish (Hill Climbing)
        # Limit polish iterations
        for _ in range(20):
            improved = False
            for i in range(N):
                # Try 8 neighbors
                for dx in [-0.02, -0.01, 0.01, 0.02]:
                    for dy in [-0.02, -0.01, 0.01, 0.02]:
                        if dx == 0 and dy == 0: continue
                        trial_c = current_c.copy()
                        trial_c[2*i] += dx
                        trial_c[2*i+1] += dy
                        trial_c[2*i] = np.clip(trial_c[2*i], 0.01, 0.99)
                        trial_c[2*i+1] = np.clip(trial_c[2*i+1], 0.01, 0.99)
                        trial_val = sum_radii(trial_c)
                        if trial_val > current_val + 1e-9:
                            current_c = trial_c
                            current_val = trial_val
                            improved = True
                            if current_val > best_val:
                                best_val = current_val
                                best_c = current_c.copy()
            if not improved:
                break
            if time.time() - start_time > time_limit:
                break

    # Final calculation
    p = best_c.reshape(N, 2)
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_neighbor_dist = np.min(dists, axis=1) / 2.0
    dist_walls = np.minimum(p, 1 - p)
    min_wall_dist = np.min(dist_walls, axis=1)
    radii = np.minimum(min_wall_dist, min_neighbor_dist)
    
    result = np.column_stack((p, radii))
    return result
