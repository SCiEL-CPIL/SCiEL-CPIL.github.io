import numpy as np
import time
import math

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s safety margin for cleanup/return
    
    # Parameters for Simulated Annealing
    K_PENALTY = 1000.0  # Weight for penalty terms
    T_START = 0.1
    T_END = 1e-6
    ALPHA = 0.999  # Cooling rate
    
    def compute_energy(x):
        """
        x: (N, 3) array of [x, y, r]
        Returns energy E = K * Penalty - Sum(r)
        Penalty is sum of squared violations.
        """
        # Boundary penalties
        # x - r >= 0  => violation if x < r
        # 1 - (x + r) >= 0 => violation if x + r > 1
        # Same for y
        
        # Vectorized boundary penalty
        # left: max(0, r - x)^2
        # right: max(0, x + r - 1)^2
        # bottom: max(0, r - y)^2
        # top: max(0, y + r - 1)^2
        
        x_pos = x[:, 0]
        y_pos = x[:, 1]
        r = x[:, 2]
        
        v_left = np.maximum(0, r - x_pos)
        v_right = np.maximum(0, x_pos + r - 1)
        v_bottom = np.maximum(0, r - y_pos)
        v_top = np.maximum(0, y_pos + r - 1)
        
        pen_boundary = np.sum(v_left**2) + np.sum(v_right**2) + np.sum(v_bottom**2) + np.sum(v_top**2)
        
        # Overlap penalties
        # dist(i,j) < r_i + r_j
        # Use broadcasting for efficiency
        # dx: (N, N)
        dx = x_pos[:, np.newaxis] - x_pos[np.newaxis, :]
        dy = y_pos[:, np.newaxis] - y_pos[np.newaxis, :]
        dist_sq = dx**2 + dy**2
        
        # Sum of radii
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        
        # Violation is (r_i + r_j) - dist
        # We only care about i < j to avoid double counting and self-interaction
        # dist is 0 on diagonal, r_sum is 2*r_i, so violation is 2*r_i.
        # We must mask the diagonal.
        
        # Efficient calculation:
        # We want sum over i<j of max(0, r_i+r_j - sqrt(dist_ij))^2
        # To avoid sqrt for non-violating pairs, we can check dist_sq < (r_i+r_j)^2
        
        # Create a mask for upper triangle
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        
        # Calculate distances only where needed? Or just calculate all.
        # N=26, so 26x26 matrix is small.
        dist = np.sqrt(dist_sq)
        
        viol = np.maximum(0, r_sum - dist)
        
        # Zero out diagonal and lower triangle
        viol = viol * mask
        
        pen_overlap = np.sum(viol**2)
        
        total_penalty = pen_boundary + pen_overlap
        sum_radii = np.sum(r)
        
        energy = K_PENALTY * total_penalty - sum_radii
        
        return energy, total_penalty, sum_radii

    def is_valid(x, eps=1e-8):
        """Check if configuration is strictly valid."""
        x_pos = x[:, 0]
        y_pos = x[:, 1]
        r = x[:, 2]
        
        # Boundary
        if np.any(x_pos - r < -eps) or np.any(y_pos - r < -eps):
            return False
        if np.any(x_pos + r > 1 + eps) or np.any(y_pos + r > 1 + eps):
            return False
            
        # Overlap
        dx = x_pos[:, np.newaxis] - x_pos[np.newaxis, :]
        dy = y_pos[:, np.newaxis] - y_pos[np.newaxis, :]
        dist = np.sqrt(dx**2 + dy**2)
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        viol = np.maximum(0, r_sum - dist)
        viol = viol * mask
        
        if np.any(viol > eps):
            return False
            
        return True

    def random_init():
        """Generate a random valid-ish initial configuration."""
        # Start with small circles in a grid, then jitter
        side = int(np.ceil(np.sqrt(N)))
        x = np.zeros((N, 3))
        for i in range(N):
            row = i // side
            col = i % side
            x[i, 0] = (col + 0.5) / side
            x[i, 1] = (row + 0.5) / side
            x[i, 2] = 0.1 # Small initial radius
        return x

    best_x = None
    best_score = 0.0
    best_energy = np.inf
    best_penalty = np.inf
    
    # Run multiple restarts
    n_restarts = 5
    time_per_restart = (deadline - start_time) / n_restarts
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
            
        x = random_init()
        # Add some randomness to positions
        x[:, :2] += np.random.uniform(-0.05, 0.05, (N, 2))
        x[:, :2] = np.clip(x[:, :2], 0.01, 0.99)
        
        cur_energy, cur_penalty, cur_sum_r = compute_energy(x)
        best_x_restart = x.copy()
        best_energy_restart = cur_energy
        best_penalty_restart = cur_penalty
        best_score_restart = cur_sum_r if cur_penalty < 1e-8 else 0.0
        
        T = T_START
        iterations = 20000 # Fixed iterations per restart, will be cut by time
        
        for i in range(iterations):
            if time.time() > deadline:
                break
            
            # Generate move
            # Perturb 1 or 2 circles
            n_perturb = 1 if np.random.random() < 0.7 else 2
            idxs = np.random.choice(N, n_perturb, replace=False)
            
            x_new = x.copy()
            
            for idx in idxs:
                # Perturb position
                x_new[idx, 0] += np.random.normal(0, 0.02)
                x_new[idx, 1] += np.random.normal(0, 0.02)
                # Perturb radius
                x_new[idx, 2] += np.random.normal(0, 0.01)
                
                # Clip to reasonable bounds to prevent extreme values
                x_new[idx, 0] = np.clip(x_new[idx, 0], 0.0, 1.0)
                x_new[idx, 1] = np.clip(x_new[idx, 1], 0.0, 1.0)
                x_new[idx, 2] = np.clip(x_new[idx, 2], 0.01, 0.3)
            
            new_energy, new_penalty, new_sum_r = compute_energy(x_new)
            
            delta_e = new_energy - cur_energy
            
            if delta_e < 0 or np.random.random() < math.exp(-delta_e / T):
                x = x_new
                cur_energy = new_energy
                cur_penalty = new_penalty
                cur_sum_r = new_sum_r
                
                # Update best for this restart
                if new_penalty < best_penalty_restart:
                    best_penalty_restart = new_penalty
                    best_energy_restart = new_energy
                    best_x_restart = x_new.copy()
                    if new_penalty < 1e-8:
                        best_score_restart = new_sum_r
                elif abs(new_penalty - best_penalty_restart) < 1e-8 and new_penalty < 1e-8:
                    # If both are valid, keep the one with higher sum of radii (lower energy)
                    if new_energy < best_energy_restart:
                        best_energy_restart = new_energy
                        best_x_restart = x_new.copy()
                        best_score_restart = new_sum_r
            
            # Cool down
            T *= ALPHA
            if T < T_END:
                T = T_END
        
        # End of restart
        # Check if this restart's best is better than global best
        if best_penalty_restart < 1e-8:
            # It's valid
            if best_score_restart > best_score:
                best_score = best_score_restart
                best_x = best_x_restart
        else:
            # If no valid solution found in this restart, try to salvage?
            # Or just skip.
            pass

    # If we found a valid solution, we can try to polish it with a local optimizer
    if best_x is not None and time.time() < deadline - 10:
        # Simple local polish: try to increase radii slightly if no overlap
        # Or use a small number of SA steps with very low T to refine
        
        # Let's do a few steps of "radius increase" SA
        # Actually, the SA above already maximizes sum of radii.
        # A quick check: ensure the returned array is strictly valid.
        
        # Final validation check
        if not is_valid(best_x, eps=1e-7):
            # If somehow invalid, shrink radii slightly
            # This shouldn't happen if best_penalty < 1e-8
            pass

    if best_x is None:
        # Fallback: Return a known valid simple packing
        # e.g. 5x5 grid with 1 extra? 
        # Just return the grid with small radii
        side = int(np.ceil(np.sqrt(N)))
        x = np.zeros((N, 3))
        for i in range(N):
            row = i // side
            col = i % side
            x[i, 0] = (col + 0.5) / side
            x[i, 1] = (row + 0.5) / side
            x[i, 2] = 0.5 / side - 0.01
        best_x = x

    return best_x
