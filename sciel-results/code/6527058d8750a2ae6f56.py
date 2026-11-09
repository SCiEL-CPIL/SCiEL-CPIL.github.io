"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    def calc_radius(p):
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        wall_dists = 0.5 - np.max(np.abs(p), axis=1)
        min_wall_dist = np.min(wall_dists)
        r = min(min_pair_dist / 2.0, min_wall_dist)
        min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
        min_wall_idx = np.argmin(wall_dists)
        return r, min_pair_idx, min_wall_idx

    def sa_polish(pos, time_limit):
        """Run SA and return best position."""
        current_r, current_min_pair, current_min_wall = calc_radius(pos)
        best_r = current_r
        best_pos = pos.copy()
        
        T = 0.005
        T_end = 1e-6
        # Estimate steps based on time
        steps = 80000
        
        alpha = (T_end / T) ** (1.0 / steps)
        current_pos = pos.copy()
        
        for step in range(steps):
            if time.time() > time_limit:
                break
                
            i = np.random.randint(N)
            
            move_size = 0.02 * (T / 0.005)
            move = np.random.uniform(-move_size, move_size, 2)
            
            new_pos = current_pos.copy()
            new_pos[i] += move
            
            if np.any(new_pos[i] > 0.5) or np.any(new_pos[i] < -0.5):
                continue
            
            # Fast check
            # If i is not in min pair/wall, we can be faster
            i_in_pair = (current_min_pair[0] == i or current_min_pair[1] == i)
            i_is_wall = (current_min_wall == i)
            
            if i_in_pair or i_is_wall:
                new_r, new_min_pair, new_min_wall = calc_radius(new_pos)
            else:
                dists_i = np.sqrt(np.sum((new_pos[i] - current_pos)**2, axis=1))
                dists_i[i] = np.inf
                min_pair_dist_new = np.min(dists_i)
                wall_dist_i = 0.5 - np.max(np.abs(new_pos[i]))
                
                new_r = min(current_r, min_pair_dist_new / 2.0, wall_dist_i)
                
                if new_r < current_r:
                    if min_pair_dist_new / 2.0 < current_r and min_pair_dist_new / 2.0 <= wall_dist_i:
                        j = np.argmin(dists_i)
                        new_min_pair = (i, j) if i < j else (j, i)
                        new_min_wall = current_min_wall
                    elif wall_dist_i < current_r:
                        new_min_wall = i
                        new_min_pair = current_min_pair
                    else:
                        new_min_pair = current_min_pair
                        new_min_wall = current_min_wall
                else:
                    new_min_pair = current_min_pair
                    new_min_wall = current_min_wall
            
            delta_r = new_r - current_r
            if delta_r > 0 or np.random.rand() < np.exp(delta_r / T):
                current_pos = new_pos
                current_r = new_r
                current_min_pair = new_min_pair
                current_min_wall = new_min_wall
                
                if current_r > best_r:
                    best_r = current_r
                    best_pos = current_pos.copy()
            
            T *= alpha
            
        return best_pos

    def lbfgs_polish(pos, time_limit):
        """Run L-BFGS-B to polish the position."""
        if time.time() > time_limit:
            return pos
            
        # Flatten
        x0 = pos.flatten()
        
        def obj(flat):
            p = flat.reshape(N, 2)
            diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            min_pair_dist = np.min(dists)
            wall_dists = 0.5 - np.max(np.abs(p), axis=1)
            min_wall_dist = np.min(wall_dists)
            r = min(min_pair_dist / 2.0, min_wall_dist)
            return -r
        
        # Constraints: bounds
        bounds = [(-0.5, 0.5)] * (2 * N)
        
        try:
            res = minimize(
                obj, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 800, 'maxfun': 8000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            return res.x.reshape(N, 2)
        except Exception:
            return pos

    def bisection_polish(pos, time_limit):
        """Use bisection on radius with penalty-based feasibility to refine."""
        # Start from current radius
        current_r, _, _ = calc_radius(pos)
        
        # We want to find the max r such that there exists a configuration with all distances >= 2r and wall dists >= r
        # Actually, for a fixed configuration, r is determined. 
        # Bisection is more useful if we can move points.
        # Let's use a penalty method: minimize sum of violations + lambda * (r_target - actual_r)^2
        # Or simpler: try to increase r slightly and see if we can fix violations.
        
        # Let's stick to a more direct approach: 
        # 1. Get current r.
        # 2. Set target_r = r * (1 + 1e-6)
        # 3. Run a constrained optimization to try to achieve target_r.
        # 4. If successful, update r and repeat.
        
        best_pos = pos.copy()
        best_r = current_r
        
        # Simple local search with increasing radius target
        # This is essentially what SA does, but let's try a specific "push" strategy.
        
        # Identify the critical pairs/walls
        # Try to move the most constrained points to increase the min distance.
        
        # This function is a placeholder for now, as the SA + LBFGS pipeline seems robust.
        # If we have time, we can implement a more sophisticated local search.
        
        return best_pos

    # 1. Initialization
    np.random.seed(42)
    
    best_overall_r = -np.inf
    best_overall_pos = None
    
    # Generate several diverse initial configurations
    configs = []
    
    # Grid
    m = 8
    n = 7
    ticks_x = (np.arange(m) + 0.5) / m - 0.5
    ticks_y = (np.arange(n) + 0.5) / n - 0.5
    grid = np.array([(x, y) for y in ticks_y for x in ticks_x])
    pos_grid = grid[:N].copy()
    pos_grid += np.random.uniform(-0.01, 0.01, pos_grid.shape)
    pos_grid = np.clip(pos_grid, -0.5, 0.5)
    configs.append(pos_grid)
    
    # Random
    pos_rand = np.random.uniform(-0.4, 0.4, (N, 2))
    configs.append(pos_rand)
    
    # Honeycomb-like
    # Hexagonal packing
    pos_hex = np.zeros((N, 2))
    idx = 0
    for row in range(10):
        for col in range(10):
            if idx >= N: break
            x = col * 0.11 - 0.5
            y = row * 0.095 - 0.5
            if row % 2 == 1:
                x += 0.055
            pos_hex[idx] = [x, y]
            idx += 1
    pos_hex += np.random.uniform(-0.005, 0.005, pos_hex.shape)
    pos_hex = np.clip(pos_hex, -0.5, 0.5)
    configs.append(pos_hex)
    
    # Run optimization on each config
    for i, config in enumerate(configs):
        if time.time() > deadline - 10:
            break
            
        # SA Phase
        # Allocate time based on remaining time
        sa_time_limit = time.time() + 35.0
        pos_sa = sa_polish(config, sa_time_limit)
        
        # L-BFGS Phase
        lbfgs_time_limit = time.time() + 15.0
        pos_final = lbfgs_polish(pos_sa, lbfgs_time_limit)
        
        r, _, _ = calc_radius(pos_final)
        
        if r > best_overall_r:
            best_overall_r = r
            best_overall_pos = pos_final.copy()
            
        # If we have time, do a second round of SA + LBFGS on the best result
        if time.time() < deadline - 20:
            # Refine the best result further
            sa_time_limit2 = time.time() + 25.0
            pos_sa2 = sa_polish(best_overall_pos, sa_time_limit2)
            
            lbfgs_time_limit2 = time.time() + 15.0
            pos_final2 = lbfgs_polish(pos_sa2, lbfgs_time_limit2)
            
            r2, _, _ = calc_radius(pos_final2)
            if r2 > best_overall_r:
                best_overall_r = r2
                best_overall_pos = pos_final2.copy()
                
    if best_overall_pos is None:
        # Fallback
        return configs[0]
        
    return best_overall_pos
# EVOLVE-BLOCK-END
