import numpy as np
import time

N = 50

def _score(centers):
    """Compute the radius for a given set of centers."""
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    wall_dists = np.min(np.abs(centers), axis=1)
    wall_min = np.min(wall_dists)
    from scipy.spatial.distance import pdist
    if N > 1:
        pair_dists = pdist(centers)
        pair_min = np.min(pair_dists) / 2.0
    else:
        pair_min = np.inf
    return min(wall_min, pair_min)

def _initial_guess():
    """Generate a structured initial guess."""
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    idx = np.random.choice(len(grid), N, replace=False)
    centers = grid[idx].copy()
    centers += np.random.normal(0, 0.01, centers.shape)
    return centers

def construct():
    t_start = time.time()
    deadline = t_start + 145.0
    
    np.random.seed(42)
    centers = _initial_guess()
    
    best_centers = centers.copy()
    best_r = _score(centers)
    
    current_r = 0.1
    r_step = 0.002
    
    # Simulated Annealing parameters
    T = 0.05
    T_min = 1e-8
    cooling = 0.9995
    
    while time.time() < deadline:
        # Run SA for current_r
        T = 0.05
        # Number of iterations depends on time left
        # Estimate time per iteration: ~50us? 1000 iters = 0.05s
        # Let's do chunks of 1000 iterations
        
        for _ in range(1000):
            if time.time() > deadline:
                break
                
            i = np.random.randint(N)
            step = np.random.normal(0, 0.05, 3)
            new_centers = centers.copy()
            new_centers[i] += step
            new_centers[i] = np.clip(new_centers[i], -0.5, 0.5)
            
            # Calculate delta cost for current_r
            old_wall_i = np.maximum(0, current_r - np.min(np.abs(centers[i])))
            old_wall_cost_i = old_wall_i ** 2
            
            old_dists_i = np.linalg.norm(centers[i] - centers, axis=1)
            old_pair_viol_i = np.maximum(0, 2*current_r - old_dists_i)
            old_pair_cost_i = np.sum(old_pair_viol_i ** 2)
            
            new_wall_i = np.maximum(0, current_r - np.min(np.abs(new_centers[i])))
            new_wall_cost_i = new_wall_i ** 2
            
            new_dists_i = np.linalg.norm(new_centers[i] - centers, axis=1)
            new_pair_viol_i = np.maximum(0, 2*current_r - new_dists_i)
            new_pair_cost_i = np.sum(new_pair_viol_i ** 2)
            
            delta_cost = (new_wall_cost_i + new_pair_cost_i) - (old_wall_cost_i + old_pair_cost_i)
            
            if delta_cost < 0 or np.random.random() < np.exp(-delta_cost / T):
                centers = new_centers
                
            T *= cooling
            
            if T < T_min:
                T = T_min
                
        # After 1000 steps, check if we can increase r
        # Calculate full cost
        # This is O(N^2) but only done every 1000 steps
        wall_dists = np.min(np.abs(centers), axis=1)
        wall_viol = np.maximum(0, current_r - wall_dists)
        wall_cost = np.sum(wall_viol ** 2)
        
        dists = np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_viol = np.maximum(0, 2*current_r - dists)
        pair_cost = np.sum(pair_viol ** 2)
        
        total_cost = wall_cost + pair_cost
        
        if total_cost < 1e-8:
            # Valid
            current_r += r_step
            # Update best
            score = _score(centers)
            if score > best_r:
                best_r = score
                best_centers = centers.copy()
        else:
            # If cost is high, we might want to keep cooling at this r
            # But if we are stuck, we might want to restart?
            # For now, just continue.
            pass

    return best_centers
