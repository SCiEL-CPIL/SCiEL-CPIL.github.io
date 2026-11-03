import numpy as np
import time

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    best_score = -1
    best_result = None
    
    def validate(x, y, r):
        # Check bounds
        if np.any(x < r - 1e-9) or np.any(x > 1 - r + 1e-9):
            return False
        if np.any(y < r - 1e-9) or np.any(y > 1 - r + 1e-9):
            return False
        # Check overlaps
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)
        overlap = r[:, None] + r[None, :] - dist
        if np.max(overlap) > 1e-7:
            return False
        return True

    # Define initial configurations
    initializations = []
    
    # 5x5 Grid
    side = 5
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            cells.append([x, y])
    base = np.array(cells[:N])
    initializations.append(base)
    
    # Jittered versions
    for k in range(20):
        jittered = base.copy()
        jittered[:, 0] += np.random.uniform(-0.1, 0.1, N)
        jittered[:, 1] += np.random.uniform(-0.1, 0.1, N)
        jittered = np.clip(jittered, 0.1, 0.9)
        initializations.append(jittered)
        
    # Random configuration
    rand_x = np.random.uniform(0.2, 0.8, N)
    rand_y = np.random.uniform(0.2, 0.8, N)
    initializations.append(np.column_stack((rand_x, rand_y)))

    for init in initializations:
        if time.time() > deadline:
            break
            
        x = init[:, 0].copy()
        y = init[:, 1].copy()
        r = np.full(N, 0.05) # Start small
        
        # Optimize
        while time.time() < deadline:
            # Try to increase radii
            delta = 0.01
            new_r = r + delta
            
            # Check boundary constraint: r <= 0.5
            max_possible = 0.5 - np.max(r)
            if max_possible <= 0:
                break
            if delta > max_possible:
                delta = max_possible
                new_r = r + delta
                if delta < 1e-6:
                    break
            
            # Resolve overlaps with gradient descent on penalty
            success = True
            # We use a fixed number of iterations for push
            for _ in range(50):
                # Calculate distances
                dx = x[:, None] - x[None, :]
                dy = y[:, None] - y[None, :]
                dist = np.sqrt(dx**2 + dy**2)
                np.fill_diagonal(dist, np.inf)
                
                # Overlaps
                overlap = new_r[:, None] + new_r[None, :] - dist
                # Only consider positive overlaps
                overlap[overlap < 0] = 0
                
                if np.max(overlap) < 1e-9:
                    break # No overlaps
                
                # Gradient of penalty P = 0.5 * sum(overlap^2)
                # dP/dx_i = sum_j overlap_ij * (x_i - x_j) / dist_ij
                grad_x = np.zeros(N)
                grad_y = np.zeros(N)
                
                # Vectorized gradient calculation
                mask = overlap > 0
                # We need to divide by dist, handle dist=0
                safe_dist = np.where(dist < 1e-6, 1e-6, dist)
                
                # Factor for each pair
                factor = overlap / safe_dist
                np.fill_diagonal(factor, 0)
                
                # Sum over j for each i
                sum_factor = np.sum(factor, axis=1)
                weighted_x = np.sum(factor * x[None, :], axis=1)
                weighted_y = np.sum(factor * y[None, :], axis=1)
                
                grad_x = x * sum_factor - weighted_x
                grad_y = y * sum_factor - weighted_y
                
                # Step
                step = 0.5
                x -= step * grad_x
                y -= step * grad_y
                
                # Clamp to bounds defined by new_r
                x = np.clip(x, new_r, 1 - new_r)
                y = np.clip(y, new_r, 1 - new_r)
            else:
                # If we didn't break, we might still have overlaps
                # Check final overlap
                dx = x[:, None] - x[None, :]
                dy = y[:, None] - y[None, :]
                dist = np.sqrt(dx**2 + dy**2)
                np.fill_diagonal(dist, np.inf)
                final_overlap = np.max(new_r[:, None] + new_r[None, :] - dist)
                if final_overlap > 1e-6:
                    success = False
            
            if success:
                r = new_r
                # Add small random perturbation to escape local minima
                if np.random.random() < 0.05:
                    x += np.random.normal(0, 0.005, N)
                    y += np.random.normal(0, 0.005, N)
                    x = np.clip(x, r, 1 - r)
                    y = np.clip(y, r, 1 - r)
            else:
                # Reduce delta
                delta /= 2
                if delta < 1e-5:
                    break
        
        # Calculate score
        score = np.sum(r)
        if score > best_score:
            if validate(x, y, r):
                best_score = score
                best_result = np.column_stack((x, y, r))
            else:
                # If invalid, try to shrink radii slightly to make valid?
                # Or just skip this result
                pass
            
    if best_result is None:
        # Fallback to simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_result
