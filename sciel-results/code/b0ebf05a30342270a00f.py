import numpy as np
import time

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    best_score = 0.0
    best_config = None
    
    # Multiple restarts with diverse initializations
    num_restarts = 30
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize with different strategies
        np.random.seed(42 + restart * 7)
        
        if restart % 3 == 0:
            # Jittered grid
            side = int(np.ceil(np.sqrt(N)))
            xs = []
            ys = []
            rs = []
            for i in range(side):
                for j in range(side):
                    if len(xs) < N:
                        x = (i + 0.5) / side + np.random.uniform(-0.08, 0.08)
                        y = (j + 0.5) / side + np.random.uniform(-0.08, 0.08)
                        x = np.clip(x, 0.1, 0.9)
                        y = np.clip(y, 0.1, 0.9)
                        xs.append(x)
                        ys.append(y)
                        rs.append(0.06)
        elif restart % 3 == 1:
            # Random positions with varied radii
            xs = np.random.uniform(0.15, 0.85, N)
            ys = np.random.uniform(0.15, 0.85, N)
            rs = np.random.uniform(0.03, 0.08, N)
        else:
            # Clustered initialization - some larger, some smaller
            xs = []
            ys = []
            rs = []
            # Place 5 "large" circles first
            for _ in range(5):
                x = np.random.uniform(0.2, 0.8)
                y = np.random.uniform(0.2, 0.8)
                r = np.random.uniform(0.08, 0.12)
                xs.append(x)
                ys.append(y)
                rs.append(r)
            # Place remaining 21 smaller circles
            for _ in range(N - 5):
                x = np.random.uniform(0.1, 0.9)
                y = np.random.uniform(0.1, 0.9)
                r = np.random.uniform(0.03, 0.06)
                xs.append(x)
                ys.append(y)
                rs.append(r)
        
        x = np.array(xs, dtype=np.float64)
        y = np.array(ys, dtype=np.float64)
        r = np.array(rs, dtype=np.float64)
        
        # Gradient Descent with adaptive penalties
        lr = 0.005
        lambda_overlap = 5.0
        lambda_bound = 5.0
        
        # Determine max steps based on remaining time
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        # Estimate ~0.002s per step for N=26
        steps = int(min(8000, max(2000, remaining / 0.003)))
        
        for step in range(steps):
            if time.time() > deadline:
                break
            
            # Calculate Distances
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d = np.sqrt(dx**2 + dy**2)
            
            # Avoid division by zero in gradients
            d_safe = np.where(d < 1e-8, 1e-8, d)
            
            # Overlap Penalty
            r_sum = r[:, None] + r[None, :]
            overlap = np.maximum(0, r_sum - d)
            np.fill_diagonal(overlap, 0)
            
            # Gradients for Overlap
            grad_x_overlap = -2 * lambda_overlap * np.sum(overlap * (dx / d_safe), axis=1)
            grad_y_overlap = -2 * lambda_overlap * np.sum(overlap * (dy / d_safe), axis=1)
            grad_r_overlap = 2 * lambda_overlap * np.sum(overlap, axis=1)
            
            # Boundary Penalty
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
            
            # Adaptive learning rate: reduce when gradients are large (avoiding instability)
            grad_mag = np.sqrt(np.sum(grad_x**2) + np.sum(grad_y**2) + np.sum(grad_r**2))
            if grad_mag > 10:
                effective_lr = lr * (10 / grad_mag)
            else:
                effective_lr = lr
            
            # Update
            x -= effective_lr * grad_x
            y -= effective_lr * grad_y
            r -= effective_lr * grad_r
            
            # Clip radii to be positive and reasonable
            r = np.maximum(r, 0.01)
            r = np.minimum(r, 0.4)
            
            # Clip positions to [0,1]
            x = np.clip(x, 0, 1)
            y = np.clip(y, 0, 1)
            
            # Slow learning rate decay
            lr *= 0.9995
            
            # Adaptive penalty: increase as we get closer to valid
            if step % 500 == 0 and step > 0:
                # Check if we're making progress
                current_overlap = np.sum(overlap)
                if current_overlap < 0.01:
                    lambda_overlap *= 0.9  # Reduce overlap penalty when nearly valid
                    lambda_bound *= 0.9
                elif current_overlap > 0.5:
                    lambda_overlap *= 1.1  # Increase if lots of overlap
                    lambda_bound *= 1.1
        
        # Post-processing: Ensure validity
        # 1. Clamp radii to boundary constraints
        r_max_bound = np.minimum(
            np.minimum(x, 1 - x),
            np.minimum(y, 1 - y)
        )
        r = np.minimum(r, r_max_bound)
        
        # 2. Iteratively repair overlaps by shrinking only problematic circles
        for _ in range(200):
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d = np.sqrt(dx**2 + dy**2)
            np.fill_diagonal(d, np.inf)
            
            r_sum = r[:, None] + r[None, :]
            overlaps = d < r_sum
            np.fill_diagonal(overlaps, False)
            
            if not np.any(overlaps):
                break
            
            # Find circles involved in overlaps and shrink them
            overlap_count = np.sum(overlaps, axis=1)
            for i in range(N):
                if overlap_count[i] > 0:
                    # Shrink this circle based on how many overlaps it has
                    r[i] *= (1 - 0.02 * overlap_count[i])
            
            # Re-clamp to boundaries
            r_max_bound = np.minimum(
                np.minimum(x, 1 - x),
                np.minimum(y, 1 - y)
            )
            r = np.minimum(r, r_max_bound)
        
        # Final check for validity
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(d, np.inf)
        
        r_sum = r[:, None] + r[None, :]
        overlaps = d < r_sum
        np.fill_diagonal(overlaps, False)
        
        # Check boundaries
        bound_violations = (r > x) | (r > 1 - x) | (r > y) | (r > 1 - y)
        
        if not np.any(overlaps) and not np.any(bound_violations):
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_config = np.column_stack((x, y, r))
                
    if best_config is None:
        # Fallback to seed if optimization failed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_config
