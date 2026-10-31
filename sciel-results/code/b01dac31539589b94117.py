"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

N = 26

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s kill
    
    # We'll use a gradient-based optimization approach with random restarts
    # to pack 26 circles in the unit square, maximizing sum of radii.
    
    best_sum = 0.0
    best_config = None
    
    def compute_obj(config):
        """
        config: (n, 3) array of (x, y, r)
        Returns: objective value (sum of radii) if valid, else -inf
        Also checks validity.
        """
        x = config[:, 0]
        y = config[:, 1]
        r = config[:, 2]
        
        # Check bounds: circles must be inside [0,1]^2
        if np.any(x < 0) or np.any(x > 1) or np.any(y < 0) or np.any(y > 1):
            return -np.inf
        if np.any(r < 0):
            return -np.inf
        if np.any(x + r > 1 + 1e-9) or np.any(x - r < -1e-9) or \
           np.any(y + r > 1 + 1e-9) or np.any(y - r < -1e-9):
            return -np.inf
        
        # Check pairwise non-overlap
        n = len(config)
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j] - 1e-9:
                    return -np.inf
        
        return np.sum(r)
    
    def make_valid(config):
        """
        Ensure config is valid by adjusting positions and radii.
        Returns a valid config or None.
        """
        config = config.copy()
        # Clamp positions to keep circles inside the square
        for i in range(N):
            x, y, r = config[i]
            # Clamp radius
            r = max(0.0, min(r, 0.5))
            # Clamp position
            x = max(r, min(1.0 - r, x))
            y = max(r, min(1.0 - r, y))
            config[i] = [x, y, r]
        
        # Iteratively resolve overlaps
        for iteration in range(1000):
            resolved = True
            for i in range(N):
                for j in range(i+1, N):
                    dx = config[i,0] - config[j,0]
                    dy = config[i,1] - config[j,1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = config[i,2] + config[j,2]
                    if dist < min_dist - 1e-10:
                        resolved = False
                        if dist < 1e-10:
                            # Separate randomly
                            angle = np.random.uniform(0, 2*np.pi)
                            dx = np.cos(angle)
                            dy = np.sin(angle)
                            dist = 1e-10
                        overlap = min_dist - dist
                        # Push apart
                        push = overlap / 2 + 1e-6
                        nx = dx / dist
                        ny = dy / dist
                        config[i,0] += push * nx
                        config[i,1] += push * ny
                        config[j,0] -= push * nx
                        config[j,1] -= push * ny
                        # Re-clamp
                        for k in [i, j]:
                            r_k = config[k,2]
                            config[k,0] = max(r_k, min(1.0 - r_k, config[k,0]))
                            config[k,1] = max(r_k, min(1.0 - r_k, config[k,1]))
            if resolved:
                break
        
        # Final check
        obj = compute_obj(config)
        if np.isinf(obj):
            return None
        return config
    
    def optimize_from_initial(config_init, max_iters=5000, lr_init=0.01):
        """
        Gradient-based optimization starting from config_init.
        Uses a penalty method for constraints.
        """
        config = config_init.copy().astype(np.float64)
        n = N
        lr = lr_init
        best_local_sum = -np.inf
        best_local_config = config.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            x = config[:, 0]
            y = config[:, 1]
            r = config[:, 2]
            
            # Compute gradients
            grad_x = np.zeros(n)
            grad_y = np.zeros(n)
            grad_r = np.zeros(n)
            
            # Objective: maximize sum(r), so grad_r = 1 for all
            grad_r[:] = 1.0
            
            # Penalty for boundary violations
            for i in range(n):
                # Left boundary: x - r >= 0
                viol = r[i] - x[i]
                if viol > 0:
                    grad_x[i] -= 10 * viol
                    grad_r[i] += 10 * viol
                # Right boundary: 1 - x - r >= 0
                viol = r[i] + x[i] - 1
                if viol > 0:
                    grad_x[i] += 10 * viol
                    grad_r[i] += 10 * viol
                # Bottom boundary: y - r >= 0
                viol = r[i] - y[i]
                if viol > 0:
                    grad_y[i] -= 10 * viol
                    grad_r[i] += 10 * viol
                # Top boundary: 1 - y - r >= 0
                viol = r[i] + y[i] - 1
                if viol > 0:
                    grad_y[i] += 10 * viol
                    grad_r[i] += 10 * viol
            
            # Penalty for overlaps
            for i in range(n):
                for j in range(i+1, n):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = r[i] + r[j]
                    viol = min_dist - dist
                    if viol > 0:
                        if dist < 1e-10:
                            # Random direction
                            angle = np.random.uniform(0, 2*np.pi)
                            nx = np.cos(angle)
                            ny = np.sin(angle)
                        else:
                            nx = dx / dist
                            ny = dy / dist
                        # Gradient of penalty: push apart
                        grad_x[i] += 10 * viol * nx
                        grad_y[i] += 10 * viol * ny
                        grad_x[j] -= 10 * viol * nx
                        grad_y[j] -= 10 * viol * ny
                        grad_r[i] += 10 * viol
                        grad_r[j] += 10 * viol
            
            # Update
            config[:, 0] += lr * grad_x
            config[:, 1] += lr * grad_y
            config[:, 2] += lr * grad_r
            
            # Clamp
            for i in range(n):
                r_i = config[i, 2]
                r_i = max(0.01, min(0.5, r_i))
                config[i, 0] = max(r_i, min(1.0 - r_i, config[i, 0]))
                config[i, 1] = max(r_i, min(1.0 - r_i, config[i, 1]))
                config[i, 2] = r_i
            
            # Check validity and track best
            obj = compute_obj(config)
            if not np.isinf(obj):
                if obj > best_local_sum:
                    best_local_sum = obj
                    best_local_config = config.copy()
            
            # Adaptive learning rate
            if it % 500 == 0 and it > 0:
                lr *= 0.95
            
            # Early stopping if no improvement
            if it % 100 == 0:
                if obj > -np.inf and abs(obj - best_local_sum) < 1e-6:
                    pass
        
        return best_local_config, best_local_sum
    
    # Strategy: multiple random restarts with different initial configurations
    n_restarts = 20
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Generate initial configuration
        if restart == 0:
            # Grid-based initial
            side = int(np.ceil(np.sqrt(N)))
            cells = []
            for i in range(side):
                for j in range(side):
                    cx = (i + 0.5) / side
                    cy = (j + 0.5) / side
                    r = 0.5 / side - 0.01
                    cells.append([cx, cy, r])
            config_init = np.array(cells[:N])
        elif restart == 1:
            # Random positions, small radii
            config_init = np.random.uniform(0.1, 0.9, size=(N, 2))
            config_init = np.hstack([config_init, np.full((N, 1), 0.05)])
        elif restart == 2:
            # Hexagonal-ish packing
            config_init = np.zeros((N, 3))
            r_est = 0.08
            dx = 2 * r_est
            dy = dx * np.sqrt(3) / 2
            idx = 0
            for row in range(8):
                for col in range(8):
                    if idx >= N:
                        break
                    x = 0.1 + col * dx + (0.5 * dx if row % 2 else 0)
                    y = 0.1 + row * dy
                    if x + r_est <= 1 and y + r_est <= 1:
                        config_init[idx] = [x, y, r_est]
                        idx += 1
            # Fill remaining
            while idx < N:
                config_init[idx] = [np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9), 0.05]
                idx += 1
        else:
            # Fully random
            config_init = np.random.uniform(0.1, 0.9, size=(N, 2))
            r_min = 0.03
            r_max = 0.15
            config_init = np.hstack([config_init, np.random.uniform(r_min, r_max, size=(N, 1))])
        
        # Make valid
        valid_config = make_valid(config_init)
        if valid_config is None:
            continue
        
        # Optimize
        opt_config, opt_sum = optimize_from_initial(valid_config, max_iters=3000, lr_init=0.005)
        
        if opt_sum > best_sum:
            best_sum = opt_sum
            best_config = opt_config
        
        # Check time
        if time.time() > deadline - 5:
            break
    
    # Final fallback: if we haven't found anything good, use the grid
    if best_config is None or best_sum < 0.1:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                r = 0.5 / side - 0.005
                cells.append([cx, cy, r])
        best_config = np.array(cells[:N])
        best_sum = np.sum(best_config[:, 2])
    
    # Ensure final result is valid
    final_obj = compute_obj(best_config)
    if np.isinf(final_obj):
        # Last resort: very small circles in a grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                r = 0.5 / side - 0.01
                cells.append([cx, cy, r])
        best_config = np.array(cells[:N])
    
    return best_config
# EVOLVE-BLOCK-END
