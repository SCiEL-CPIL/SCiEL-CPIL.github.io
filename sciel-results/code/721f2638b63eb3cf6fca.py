"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    # Pre-compute indices for pair checks
    i_indices, j_indices = np.triu_indices(N, k=1)
    
    def is_valid(pos, rad):
        # Check boundaries
        if np.any(pos[:, 0] - rad < -1e-10) or np.any(pos[:, 0] + rad > 1 + 1e-10):
            return False
        if np.any(pos[:, 1] - rad < -1e-10) or np.any(pos[:, 1] + rad > 1 + 1e-10):
            return False
        
        # Check overlaps
        dx = pos[i_indices, 0] - pos[j_indices, 0]
        dy = pos[i_indices, 1] - pos[j_indices, 1]
        dist_sq = dx*dx + dy*dy
        rad_sum = rad[i_indices] + rad[j_indices]
        if np.any(dist_sq < rad_sum**2 - 1e-10):
            return False
        return True
    
    def initial_placement(seed):
        np.random.seed(seed)
        positions = np.zeros((N, 2))
        radii = np.zeros(N)
        
        # Use a 5x5 grid base (25 circles) + 1 extra
        # Or just 5x5 and map 26 circles to it with one duplicate? No, distinct.
        # Let's use a 5x5 grid for the first 25, and place the 26th in the center gap?
        # The center of a 5x5 grid is occupied by (2,2).
        # Let's just use a jittered 5x5 grid for 25, and put the 26th near the center
        # but pushed out, or just use a 6x6 grid and pick 26 spots.
        
        # Simpler: 5x5 grid with jitter.
        side = 5
        for i in range(N):
            if i < 25:
                row = i // side
                col = i % side
                offset = 0.5 / side if row % 2 == 1 else 0
                positions[i, 0] = (col + offset + 0.5) / side
                positions[i, 1] = (row + 0.5) / side
                radii[i] = 0.45 / side
            else:
                # Place 26th circle in a random valid spot, or center
                # Center is (0.5, 0.5). If (2,2) is there, it's crowded.
                # Let's put it at (0.5, 0.1) roughly?
                # Better: Just add it to the grid logic by extending side?
                # Let's just place it randomly and let SA fix it.
                positions[i, 0] = np.random.uniform(0.2, 0.8)
                positions[i, 1] = np.random.uniform(0.2, 0.8)
                radii[i] = 0.1

        # Small random perturbation
        positions += np.random.uniform(-0.02, 0.02, (N, 2))
        
        # Clip to valid range
        positions[:, 0] = np.clip(positions[:, 0], 0.01, 0.99)
        positions[:, 1] = np.clip(positions[:, 1], 0.01, 0.99)
        
        # Ensure validity by shrinking if needed
        if not is_valid(positions, radii):
             scale = 1.0
             while not is_valid(positions, radii * scale) and scale > 0.01:
                 scale *= 0.9
             radii *= scale
             
        return positions, radii
    
    def local_search(pos, rad, max_iters=2000):
        """
        Simple iterative relaxation to resolve overlaps and maximize radii.
        """
        pos = pos.copy()
        rad = rad.copy()
        
        for it in range(max_iters):
            # 1. Resolve overlaps
            any_overlap = False
            for i in range(N):
                for j in range(i+1, N):
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < 1e-6:
                        pos[i, 0] += np.random.uniform(-0.01, 0.01)
                        pos[i, 1] += np.random.uniform(-0.01, 0.01)
                        any_overlap = True
                        continue
                        
                    rad_sum = rad[i] + rad[j]
                    if dist < rad_sum:
                        # Push apart
                        push = (rad_sum - dist) / 2 * 0.5
                        nx = dx / dist
                        ny = dy / dist
                        pos[i, 0] += nx * push
                        pos[i, 1] += ny * push
                        pos[j, 0] -= nx * push
                        pos[j, 1] -= ny * push
                        any_overlap = True
                        
            # 2. Try to grow radii slightly
            # This is heuristic. We try to increase the smallest radii?
            # Or just check if we can increase all by a tiny bit.
            # Let's try to increase the radius of each circle by 0.0005 if valid.
            growth = 0.0005
            for i in range(N):
                old_r = rad[i]
                rad[i] = old_r + growth
                if is_valid(pos, rad):
                    # Keep it
                    pass
                else:
                    rad[i] = old_r
                    
            if not any_overlap and it > 100:
                break

        # Final check
        if not is_valid(pos, rad):
             # Shrink to make valid
             scale = 1.0
             while not is_valid(pos, rad * scale) and scale > 0.01:
                 scale *= 0.9
             rad *= scale
             
        return pos, rad
    
    def simulated_annealing(init_pos, init_rad, time_budget, seed):
        np.random.seed(seed)
        current_pos = init_pos.copy()
        current_rad = init_rad.copy()
        
        if not is_valid(current_pos, current_rad):
            return None, None, -np.inf
        
        current_score = np.sum(current_rad)
        best_pos = current_pos.copy()
        best_rad = current_rad.copy()
        best_score = current_score
        
        # Adaptive parameters
        T = 0.05
        T_min = 0.0005
        alpha = 0.9998
        step_size_pos = 0.03
        step_size_rad = 0.03
        start_time = time.time()
        
        for it in range(10000000):
            if time.time() - start_time > time_budget:
                break
            
            # Generate a move
            i = np.random.randint(N)
            move_type = np.random.rand()
            
            new_pos = current_pos.copy()
            new_rad = current_rad.copy()
            
            if move_type < 0.6:
                # Move position
                delta = np.random.normal(0, step_size_pos, 2)
                new_pos[i] += delta
            else:
                # Change radius
                delta_r = np.random.normal(0, step_size_rad)
                new_rad[i] += delta_r
                new_rad[i] = np.clip(new_rad[i], 0.005, 0.5)
            
            # Check validity
            if is_valid(new_pos, new_rad):
                new_score = np.sum(new_rad)
                delta_score = new_score - current_score
                
                # Acceptance probability
                if delta_score > 0:
                    accept = True
                else:
                    if T > T_min:
                        accept_prob = np.exp(delta_score / T)
                        accept = np.random.rand() < accept_prob
                    else:
                        accept = False
                
                if accept:
                    current_pos = new_pos
                    current_rad = new_rad
                    current_score = new_score
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_pos = current_pos.copy()
                        best_rad = current_rad.copy()
            else:
                # If invalid, slightly reduce step size
                step_size_pos *= 0.999
                step_size_rad *= 0.999
            
            # Cool down and adapt step sizes
            T *= alpha
            if T < T_min:
                T = T_min
            step_size_pos = max(0.001, step_size_pos * 0.99999)
            step_size_rad = max(0.001, step_size_rad * 0.99999)
        
        return best_pos, best_rad, best_score
    
    # Run multiple SA runs with different seeds
    start_time = time.time()
    time_limit = 108  # Keep buffer for 120s limit
        
    best_overall_pos = None
    best_overall_rad = None
    best_overall_score = -np.inf
        
    num_runs = 5
    time_per_run = time_limit / num_runs
        
    for run in range(num_runs):
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time < 1:
            break
            
        seed = 42 + run * 1000
        init_pos, init_rad = initial_placement(seed)
            
        # Leave time for local search
        ls_time = 5
        sa_time = min(time_per_run, remaining_time) - ls_time
        if sa_time < 1:
            sa_time = max(1, min(time_per_run, remaining_time) / 2)
            ls_time = min(time_per_run, remaining_time) - sa_time
                
        pos, rad, score = simulated_annealing(init_pos, init_rad, sa_time, seed)
            
        # Apply local search
        if pos is not None:
             ls_pos, ls_rad = local_search(pos, rad, max_iters=2000)
             if is_valid(ls_pos, ls_rad):
                 ls_score = np.sum(ls_rad)
                 if ls_score > score:
                     pos = ls_pos
                     rad = ls_rad
                     score = ls_score

        if pos is not None and score > best_overall_score:
            best_overall_score = score
            best_overall_pos = pos
            best_overall_rad = rad
        
    # Final validation
    if best_overall_pos is None or not is_valid(best_overall_pos, best_overall_rad):
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(N):
            row = i // side
            col = i % side
            cells.append([(col + 0.5) / side, (row + 0.5) / side, 0.3 / side])
        return np.array(cells)
        
    return np.column_stack([best_overall_pos, best_overall_rad])
# EVOLVE-BLOCK-END
