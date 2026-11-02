"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time
import random

N = 21

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Rectangle perimeter is 4, so 2*(w+h) = 4 => w+h = 2.
    # Let w = 2 * t, h = 2 * (1-t), where t in (0, 1).
    
    def get_rect_dims(t):
        t = np.clip(t, 0.1, 0.9)
        w = 2.0 * t
        h = 2.0 * (1.0 - t)
        return w, h

    def evaluate(state):
        """
        Returns (score, penalty). Score = sum_r - penalty.
        Higher score is better. Valid solutions have penalty=0.
        Vectorized for speed.
        """
        t = state[0]
        w, h = get_rect_dims(t)
        
        # Unpack state into arrays
        coords = state[1:].reshape(N, 3)
        x = coords[:, 0]
        y = coords[:, 1]
        radii = coords[:, 2]
        
        # Ensure radii are positive
        radii_safe = np.maximum(radii, 1e-8)
        
        sum_r = np.sum(radii_safe)
        
        penalty = 0.0
        
        # Boundary violations
        # x - r < 0 => violation r - x
        # x + r > w => violation x + r - w
        # y - r < 0 => violation r - y
        # y + r > h => violation y + r - h
        
        viol_left = np.maximum(0, radii_safe - x)
        viol_right = np.maximum(0, x + radii_safe - w)
        viol_bottom = np.maximum(0, radii_safe - y)
        viol_top = np.maximum(0, y + radii_safe - h)
        
        boundary_penalty = np.sum(viol_left) + np.sum(viol_right) + np.sum(viol_bottom) + np.sum(viol_top)
        penalty += 500.0 * boundary_penalty
        
        # Overlap violations
        # Compute pairwise distances
        # dist[i, j] = ||p_i - p_j||
        # min_dist[i, j] = r_i + r_j
        
        # Vectorized pairwise distance calculation
        # x has shape (N,), x[:, None] has shape (N, 1), x[None, :] has shape (1, N)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist_sq = dx*dx + dy*dy
        dist = np.sqrt(dist_sq)
        
        min_dist = radii_safe[:, None] + radii_safe[None, :]
        
        # Overlap is positive if dist < min_dist
        overlap = np.maximum(0, min_dist - dist)
        
        # We only care about i < j to avoid double counting and self-overlap (dist=0, min_dist=2r)
        # Create a mask for upper triangle
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        overlap_penalty = np.sum(overlap[mask])
        
        penalty += 1000.0 * overlap_penalty
        
        score = sum_r - penalty
        return score, penalty

    def check_valid(state, eps=1e-7):
        """
        Check if state is valid: all circles inside rectangle, no overlaps.
        """
        t = state[0]
        w, h = get_rect_dims(t)
        
        coords = state[1:].reshape(N, 3)
        x = coords[:, 0]
        y = coords[:, 1]
        radii = coords[:, 2]
        
        if np.any(radii < 1e-6):
            return False
        
        # Boundary check
        if np.any(x - radii < -eps) or np.any(x + radii > w + eps) or \
           np.any(y - radii < -eps) or np.any(y + radii > h + eps):
            return False
        
        # Overlap check
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist_sq = dx*dx + dy*dy
        dist = np.sqrt(dist_sq)
        
        min_dist = radii[:, None] + radii[None, :]
        
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        if np.any(dist[mask] < min_dist[mask] - eps):
            return False
            
        return True

    def make_initial_hex(rows):
        """
        Generate a hexagonal packing configuration.
        Returns state vector.
        """
        pts = []
        for k, n_circles in enumerate(rows):
            y_k = k * np.sqrt(3)
            offset = 1.0 if k % 2 == 1 else 0.0
            for i in range(n_circles):
                x_k = i * 2.0 + offset
                pts.append((x_k, y_k))
        
        if len(pts) != N:
            return None
            
        pts = np.array(pts)
        
        # Center the points
        center = pts.mean(axis=0)
        pts -= center
        
        xs = pts[:, 0]
        ys = pts[:, 1]
        w_centers = xs.max() - xs.min()
        h_centers = ys.max() - ys.min()
        
        # Add diameter (2r) to get bounding box of circles
        # Here r=1, so add 2
        w_total = w_centers + 2.0
        h_total = h_centers + 2.0
        
        # Scale to fit perimeter 4
        # w + h = 2
        # s*w_total + s*h_total = 2 => s = 2 / (w_total + h_total)
        scale = 2.0 / (w_total + h_total)
        r = scale * 1.0
        w = scale * w_total
        h = scale * h_total
        
        # Transform points to final coordinates
        xs_final = pts[:, 0] * scale + w/2.0
        ys_final = pts[:, 1] * scale + h/2.0
        
        # Construct state
        state = np.zeros(1 + 3*N)
        # t = w / 2
        state[0] = w / 2.0
        
        for i in range(N):
            state[1 + 3*i] = xs_final[i]
            state[1 + 3*i + 1] = ys_final[i]
            state[1 + 3*i + 2] = r
            
        return state

    def make_initial_grid(rows, cols):
        """
        Generate a grid packing configuration.
        """
        pts = []
        for i in range(cols):
            for j in range(rows):
                if len(pts) >= N:
                    break
                pts.append((i * 2.0, j * 2.0))
        
        if len(pts) != N:
            return None

        pts = np.array(pts)
        
        center = pts.mean(axis=0)
        pts -= center
        
        xs = pts[:, 0]
        ys = pts[:, 1]
        w_centers = xs.max() - xs.min()
        h_centers = ys.max() - ys.min()
        
        w_total = w_centers + 2.0
        h_total = h_centers + 2.0
        
        scale = 2.0 / (w_total + h_total)
        r = scale * 1.0
        w = scale * w_total
        h = scale * h_total
        
        xs_final = pts[:, 0] * scale + w/2.0
        ys_final = pts[:, 1] * scale + h/2.0
        
        state = np.zeros(1 + 3*N)
        state[0] = w / 2.0
        
        for i in range(N):
            state[1 + 3*i] = xs_final[i]
            state[1 + 3*i + 1] = ys_final[i]
            state[1 + 3*i + 2] = r
            
        return state

    def local_optimize(state, time_limit):
        """
        Run simulated annealing to improve state.
        """
        best_state = state.copy()
        best_score, _ = evaluate(best_state)
        
        T_start = 0.5
        T_end = 1e-6
        # Calculate alpha based on time limit to ensure we reach T_end at time_limit
        # This is a heuristic, we'll just use a fixed alpha and check time
        alpha = 0.999
        
        T = T_start
        no_improve_count = 0
        
        while time.time() < time_limit:
            # Check time
            if time.time() > time_limit - 1:
                break
            
            # Generate neighbor
            new_state = best_state.copy()
            var_idx = random.randint(0, len(new_state)-1)
            
            if var_idx == 0:
                # Perturb aspect ratio
                new_state[0] += random.gauss(0, 0.02)
                if new_state[0] < 0.1 or new_state[0] > 0.9:
                    T *= alpha
                    if T < T_end:
                        T = T_end
                    continue
            else:
                circle_idx = (var_idx - 1) // 3
                comp = (var_idx - 1) % 3
                
                # Scale perturbation by current temperature
                # We want perturbations to be meaningful relative to the box size
                w, h = get_rect_dims(best_state[0])
                box_size = min(w, h)
                scale = T / T_start * 0.1 * box_size + 0.005 * box_size
                
                if comp == 0:  # x
                    new_state[var_idx] += random.gauss(0, scale)
                elif comp == 1:  # y
                    new_state[var_idx] += random.gauss(0, scale)
                else:  # r
                    new_state[var_idx] += random.gauss(0, scale * 0.5)
                    if new_state[var_idx] < 1e-4:
                        new_state[var_idx] = 1e-4
                    max_r = 0.5 * min(w, h)
                    if new_state[var_idx] > max_r:
                        new_state[var_idx] = max_r
            
            new_score, new_penalty = evaluate(new_state)
            
            # Acceptance criterion
            if new_score > best_score or random.random() < np.exp((new_score - best_score) / T):
                best_state = new_state
                best_score = new_score
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            T *= alpha
            if T < T_end:
                T = T_end
        
        return best_state, best_score

    # Main algorithm
    best_state = None
    best_score = -np.inf
    
    # Generate initial configurations
    initial_configs = []
    
    # Hexagonal packings
    hex_configs = [
        [7, 7, 7],
        [4, 5, 4, 4, 4],
        [3, 4, 4, 4, 4, 2],
        [2, 4, 4, 4, 4, 3],
        [5, 5, 5, 3, 3]
    ]
    
    for hc in hex_configs:
        if sum(hc) == N:
            state = make_initial_hex(hc)
            if state is not None:
                initial_configs.append(state)
                
    # Grid packings
    grid_configs = [
        (5, 5), # 25 cells, take 21
        (3, 7),
        (4, 6), # 24 cells, take 21
        (7, 3)
    ]
    
    for rc in grid_configs:
        state = make_initial_grid(rc[0], rc[1])
        if state is not None:
            initial_configs.append(state)

    # If no valid initial configs were generated (shouldn't happen), use a simple grid
    if not initial_configs:
        initial_configs.append(make_initial_grid(5, 5))

    # Shuffle to avoid bias
    random.shuffle(initial_configs)
    
    # Process each initial config
    for i, init_state in enumerate(initial_configs):
        if time.time() > deadline - 15:
            break
        
        # Time budget for this restart
        # Allocate more time to earlier restarts? Or equal?
        # Let's allocate equal time, but ensure at least 10s per restart if possible
        remaining_time = deadline - 15 - time.time()
        num_remaining = len(initial_configs) - i
        time_per_restart = max(5, remaining_time / num_remaining)
        
        sa_deadline = time.time() + time_per_restart
        
        state, score = local_optimize(init_state, sa_deadline)
        
        if score > best_score:
            best_score = score
            best_state = state.copy()
            
        # If we found a valid solution, try a quick polish
        if check_valid(best_state):
            # Try to increase radii slightly and re-optimize
            refined = best_state.copy()
            # Increase radii by 5%
            for k in range(N):
                refined[1 + 3*k + 2] *= 1.05
            # Quick local optimization
            quick_deadline = min(time.time() + 5, deadline - 5)
            refined, ref_score = local_optimize(refined, quick_deadline)
            if ref_score > best_score and check_valid(refined):
                best_score = ref_score
                best_state = refined.copy()

    # Final validity check and repair if needed
    if best_state is None:
        # Fallback: return a simple valid grid
        best_state = initial_configs[0]
    elif not check_valid(best_state):
        # Try to repair by shrinking radii
        repair = best_state.copy()
        for _ in range(50):
            if check_valid(repair):
                break
            for k in range(N):
                repair[1 + 3*k + 2] *= 0.98
        if check_valid(repair):
            best_state = repair
        else:
            # Last resort: use the first initial config
            best_state = initial_configs[0]
    
    # Extract result
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = best_state[1 + 3*i]
        result[i, 1] = best_state[1 + 3*i+1]
        result[i, 2] = best_state[1 + 3*i+2]
    
    return result

# EVOLVE-BLOCK-END
