"""Rewritten program for circle packing in unit square."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 26

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    best_sum = 0.0
    best_result = None
    
    def check_validity(x, y, r):
        """Check if packing is valid (all constraints satisfied)."""
        # Boundary check
        if np.any(x < r) or np.any(x > 1 - r) or np.any(y < r) or np.any(y > 1 - r):
            return False
        # Collision check
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                if d < r[i] + r[j] - 1e-9:
                    return False
        return True
    
    def objective(z, penalty_weight=1000.0):
        """Objective: minimize -sum(r) + penalty for violations."""
        x = z[:N]
        y = z[N:2*N]
        r = z[2*N:]
        
        # Clip radii to positive values
        r_safe = np.maximum(r, 0.001)
        
        sum_r = np.sum(r_safe)
        
        # Boundary violations
        v_bound = 0.0
        v_bound += np.sum(np.maximum(0, r_safe - x)**2)
        v_bound += np.sum(np.maximum(0, r_safe - (1 - x))**2)
        v_bound += np.sum(np.maximum(0, r_safe - y)**2)
        v_bound += np.sum(np.maximum(0, r_safe - (1 - y))**2)
        
        # Collision violations
        v_coll = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                req = r_safe[i] + r_safe[j]
                if d < req:
                    v_coll += (req - d)**2
        
        return -sum_r + penalty_weight * (v_bound + v_coll)
    
    def local_optimize(x0, y0, r0, max_iter=300):
        """Run local optimization from initial guess."""
        z0 = np.concatenate([x0, y0, r0])
        
        try:
            res = minimize(objective, z0, method='L-BFGS-B', 
                          options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-6})
            
            x_opt = res.x[:N]
            y_opt = res.x[N:2*N]
            r_opt = res.x[2*N:]
            
            # Post-process: shrink radii slightly to ensure validity
            r_opt = np.maximum(r_opt, 0.001)
            
            # Check validity and adjust if needed
            if check_validity(x_opt, y_opt, r_opt):
                return x_opt, y_opt, r_opt
            else:
                # Try to fix by shrinking radii
                scale = 0.95
                r_adjusted = r_opt * scale
                if check_validity(x_opt, y_opt, r_adjusted):
                    return x_opt, y_opt, r_adjusted
                else:
                    return None
        except Exception:
            return None
    
    def hexagonal_init(n_circles):
        """Create hexagonal packing initialization."""
        x = np.zeros(n_circles)
        y = np.zeros(n_circles)
        r = np.zeros(n_circles)
        
        # Hexagonal arrangement
        row_spacing = 0.866  # sqrt(3)/2
        col_spacing = 1.0
        
        # Estimate grid size
        cols = int(np.ceil(np.sqrt(n_circles)))
        rows = int(np.ceil(n_circles / cols))
        
        # Scale to fit in unit square
        total_width = (cols - 1) * col_spacing
        total_height = (rows - 1) * row_spacing
        
        # Add margins
        margin_x = 0.05
        margin_y = 0.05
        
        scale_x = (1 - 2*margin_x) / max(total_width, 1e-6)
        scale_y = (1 - 2*margin_y) / max(total_height, 1e-6)
        scale = min(scale_x, scale_y, 1.0)
        
        for i in range(n_circles):
            row = i // cols
            col = i % cols
            x[i] = margin_x + col * col_spacing * scale + (row % 2) * 0.5 * col_spacing * scale
            y[i] = margin_y + row * row_spacing * scale
            r[i] = 0.5 * min(col_spacing, row_spacing) * scale * 0.95
        
        # Ensure radii don't exceed bounds
        for i in range(n_circles):
            r[i] = min(r[i], min(x[i], 1-x[i], y[i], 1-y[i]) * 0.99)
        
        return x, y, r
    
    def random_init():
        """Random initialization."""
        r = np.random.uniform(0.03, 0.1, N)
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        return x, y, r
    
    def grid_init():
        """Grid initialization."""
        side = int(np.ceil(np.sqrt(N)))
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        for i in range(N):
            row = i // side
            col = i % side
            x[i] = (col + 0.5) / side
            y[i] = (row + 0.5) / side
            r[i] = 0.5 / side * 0.95
        
        return x, y, r
    
    # Multi-start optimization with different initializations
    n_hex_restarts = 20
    n_random_restarts = 30
    n_grid_restarts = 5
    
    # Hexagonal starts (likely to find good local optima)
    for k in range(n_hex_restarts):
        if time.time() > deadline:
            break
        
        # Perturb hexagonal init
        x0, y0, r0 = hexagonal_init(N)
        x0 += np.random.uniform(-0.02, 0.02, N)
        y0 += np.random.uniform(-0.02, 0.02, N)
        r0 *= np.random.uniform(0.9, 1.1, N)
        
        result = local_optimize(x0, y0, r0)
        if result is not None:
            x_opt, y_opt, r_opt = result
            current_sum = np.sum(r_opt)
            if current_sum > best_sum:
                best_sum = current_sum
                best_result = np.column_stack((x_opt, y_opt, r_opt))
    
    # Random starts
    for k in range(n_random_restarts):
        if time.time() > deadline:
            break
        
        x0, y0, r0 = random_init()
        result = local_optimize(x0, y0, r0)
        if result is not None:
            x_opt, y_opt, r_opt = result
            current_sum = np.sum(r_opt)
            if current_sum > best_sum:
                best_sum = current_sum
                best_result = np.column_stack((x_opt, y_opt, r_opt))
    
    # Grid starts
    for k in range(n_grid_restarts):
        if time.time() > deadline:
            break
        
        x0, y0, r0 = grid_init()
        x0 += np.random.uniform(-0.01, 0.01, N)
        y0 += np.random.uniform(-0.01, 0.01, N)
        
        result = local_optimize(x0, y0, r0)
        if result is not None:
            x_opt, y_opt, r_opt = result
            current_sum = np.sum(r_opt)
            if current_sum > best_sum:
                best_sum = current_sum
                best_result = np.column_stack((x_opt, y_opt, r_opt))
    
    # Fallback to seed if no better solution found
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_result
# EVOLVE-BLOCK-END
