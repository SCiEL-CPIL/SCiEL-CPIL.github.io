import numpy as np
import time

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    best_score = 0.0
    best_config = None
    
    # Helper to check validity and return penalty (0 if valid)
    def validity_penalty(x, y, r):
        n = len(r)
        # Check bounds
        if np.any(r < 0):
            return 1e10
        if np.any(x - r < -1e-9) or np.any(y - r < -1e-9):
            return 1e10
        if np.any(x + r > 1 + 1e-9) or np.any(y + r > 1 + 1e-9):
            return 1e10
        
        # Check overlaps
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j] - 1e-9:
                    return 1e10
        return 0.0

    # Helper to repair configuration by shrinking radii if invalid
    def repair_config(x, y, r):
        # First, clamp to boundaries
        r_max_bound = np.minimum(
            np.minimum(x, 1 - x),
            np.minimum(y, 1 - y)
        )
        r = np.minimum(r, r_max_bound)
        r = np.maximum(r, 0.0)
        
        # Iteratively shrink if overlaps persist
        for _ in range(100):
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d = np.sqrt(dx**2 + dy**2)
            np.fill_diagonal(d, np.inf)
            
            r_sum = r[:, None] + r[None, :]
            overlaps = d < r_sum - 1e-9
            np.fill_diagonal(overlaps, False)
            
            if not np.any(overlaps):
                return x, y, r
            
            # Shrink all radii slightly
            r *= 0.98
        
        return x, y, r

    # Gradient Descent Optimizer with adaptive learning rate
    def optimize_gradient(x, y, r, steps=2000, start_lr=0.02):
        lr = start_lr
        lambda_overlap = 15.0
        lambda_bound = 15.0
        prev_score = -np.sum(r)
        
        for step in range(steps):
            if time.time() > deadline:
                break
            
            # Calculate Distances
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d = np.sqrt(dx**2 + dy**2)
            d_safe = np.where(d < 1e-8, 1e-8, d)
            
            # Overlap Penalty Gradient
            r_sum = r[:, None] + r[None, :]
            overlap = np.maximum(0, r_sum - d)
            np.fill_diagonal(overlap, 0)
            
            grad_x_overlap = -2 * lambda_overlap * np.sum(overlap * (dx / d_safe), axis=1)
            grad_y_overlap = -2 * lambda_overlap * np.sum(overlap * (dy / d_safe), axis=1)
            grad_r_overlap = 2 * lambda_overlap * np.sum(overlap, axis=1)
            
            # Boundary Penalty Gradient
            bound_x_left = np.maximum(0, r - x)
            bound_x_right = np.maximum(0, r - (1 - x))
            bound_y_bottom = np.maximum(0, r - y)
            bound_y_top = np.maximum(0, r - (1 - y))
            
            grad_x_bound = -2 * lambda_bound * bound_x_left + 2 * lambda_bound * bound_x_right
            grad_y_bound = -2 * lambda_bound * bound_y_bottom + 2 * lambda_bound * bound_y_top
            grad_r_bound = 2 * lambda_bound * (bound_x_left + bound_x_right + bound_y_bottom + bound_y_top)
            
            # Objective: Maximize Sum(r) -> Minimize -Sum(r)
            grad_r_obj = -np.ones(N)
            
            # Total Gradients
            grad_x = grad_x_overlap + grad_x_bound
            grad_y = grad_y_overlap + grad_y_bound
            grad_r = grad_r_overlap + grad_r_bound + grad_r_obj
            
            # Update
            x_new = x - lr * grad_x
            y_new = y - lr * grad_y
            r_new = r - lr * grad_r
            
            # Clip
            r_new = np.maximum(r_new, 0.001)
            x_new = np.clip(x_new, 0, 1)
            y_new = np.clip(y_new, 0, 1)
            
            # Check if this improves the objective (sum of radii)
            current_score = np.sum(r_new)
            if current_score > prev_score:
                x, y, r = x_new, y_new, r_new
                prev_score = current_score
                # Increase learning rate slightly if improving
                lr = min(lr * 1.05, 0.05)
            else:
                # Decrease learning rate if not improving
                lr *= 0.9
            
            # Decay learning rate
            lr *= 0.998
            
        return x, y, r

    # Local Search Refinement with larger perturbations
    def local_search(x, y, r, max_iter=800):
        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        best_val = -np.sum(best_r)
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
                
            # Perturb a random circle
            idx = np.random.randint(N)
            candidate_x = best_x.copy()
            candidate_y = best_y.copy()
            candidate_r = best_r.copy()
            
            # Larger perturbations for better exploration
            scale = 0.03 if it < max_iter // 2 else 0.01
            dx = np.random.normal(0, scale)
            dy = np.random.normal(0, scale)
            dr = np.random.normal(0, scale * 0.5)
            
            candidate_x[idx] += dx
            candidate_y[idx] += dy
            candidate_r[idx] += dr
            
            # Clamp
            candidate_x[idx] = np.clip(candidate_x[idx], 0, 1)
            candidate_y[idx] = np.clip(candidate_y[idx], 0, 1)
            candidate_r[idx] = np.clip(candidate_r[idx], 0.001, 0.5)
            
            # Check validity
            pen = validity_penalty(candidate_x, candidate_y, candidate_r)
            if pen == 0:
                val = -np.sum(candidate_r)
                if val < best_val:
                    best_x, best_y, best_r = candidate_x, candidate_y, candidate_r
                    best_val = val
        
        return best_x, best_y, best_r

    # Multiple restarts with varied initialization strategies
    num_restarts = 20
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        np.random.seed(42 + restart * 7)
        
        # Different initialization strategies
        strategy = restart % 3
        
        if strategy == 0:
            # Grid-like initialization with jitter
            side = int(np.ceil(np.sqrt(N)))
            xs = []
            ys = []
            rs = []
            
            indices = list(range(side * side))
            np.random.shuffle(indices)
            
            for idx in indices:
                if len(xs) >= N:
                    break
                i = idx % side
                j = idx // side
                x = (i + 0.5) / side + np.random.uniform(-0.08, 0.08)
                y = (j + 0.5) / side + np.random.uniform(-0.08, 0.08)
                x = np.clip(x, 0.05, 0.95)
                y = np.clip(y, 0.05, 0.95)
                xs.append(x)
                ys.append(y)
                rs.append(0.06)
                
        elif strategy == 1:
            # Random initialization with larger initial radii
            xs = np.random.uniform(0.1, 0.9, N)
            ys = np.random.uniform(0.1, 0.9, N)
            rs = np.full(N, 0.05)
            xs = list(xs)
            ys = list(ys)
            rs = list(rs)
            
        else:
            # Hybrid: some grid, some random
            side = int(np.ceil(np.sqrt(N)))
            xs = []
            ys = []
            rs = []
            
            # First half from grid
            for i in range(side):
                for j in range(side):
                    if len(xs) >= N // 2:
                        break
                    x = (i + 0.5) / side + np.random.uniform(-0.05, 0.05)
                    y = (j + 0.5) / side + np.random.uniform(-0.05, 0.05)
                    x = np.clip(x, 0.05, 0.95)
                    y = np.clip(y, 0.05, 0.95)
                    xs.append(x)
                    ys.append(y)
                    rs.append(0.06)
                if len(xs) >= N // 2:
                    break
            
            # Second half random
            while len(xs) < N:
                x = np.random.uniform(0.1, 0.9)
                y = np.random.uniform(0.1, 0.9)
                xs.append(x)
                ys.append(y)
                rs.append(0.05)
        
        x = np.array(xs)
        y = np.array(ys)
        r = np.array(rs)
        
        # Run Gradient Descent with more steps
        x, y, r = optimize_gradient(x, y, r, steps=3000, start_lr=0.02)
        
        # Repair
        x, y, r = repair_config(x, y, r)
        
        # Local Search with more iterations
        x, y, r = local_search(x, y, r, max_iter=500)
        
        # Final Repair
        x, y, r = repair_config(x, y, r)
        
        # Check Validity
        if validity_penalty(x, y, r) == 0:
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_config = np.column_stack((x, y, r))
        
        # If time is running out, try a quick polish on the current best
        if time.time() > deadline - 15:
            break

    if best_config is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_config
