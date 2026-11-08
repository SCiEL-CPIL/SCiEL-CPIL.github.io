"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    """Pack 26 circles in unit square maximizing sum of radii."""
    N = 26
    
    def compute_objective(params):
        """Compute negative sum of radii (to minimize) with penalty for violations."""
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:]
        
        # Penalty for out-of-bounds
        penalty = 0.0
        for i in range(N):
            if x[i] - r[i] < 0:
                penalty += (x[i] - r[i]) ** 2
            if x[i] + r[i] > 1:
                penalty += (x[i] + r[i] - 1) ** 2
            if y[i] - r[i] < 0:
                penalty += (y[i] - r[i]) ** 2
            if y[i] + r[i] > 1:
                penalty += (y[i] + r[i] - 1) ** 2
        
        # Penalty for overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2
        
        return -np.sum(r) + 100 * penalty
    
    def is_valid(circles_arr, tol=1e-6):
        """Check if all circles are non-overlapping and within [0,1]^2."""
        n = len(circles_arr)
        for i in range(n):
            x, y, r = circles_arr[i]
            if x - r < -tol or x + r > 1 + tol or y - r < -tol or y + r > 1 + tol:
                return False
        for i in range(n):
            for j in range(i+1, n):
                xi, yi, ri = circles_arr[i]
                xj, yj, rj = circles_arr[j]
                dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                if dist < ri + rj - tol:
                    return False
        return True
    
    # Generate multiple initial configurations and pick the best
    best_result = None
    best_score = -np.inf
    
    np.random.seed(42)
    
    # Strategy 1: Hexagonal-like grid
    row_configs = [5, 6, 5, 6, 4]
    n_rows = len(row_configs)
    circles_list = []
    for i, n_in_row in enumerate(row_configs):
        x_positions = [(j + 0.5) / n_in_row for j in range(n_in_row)]
        y_pos = (i + 0.5) / n_rows
        for x in x_positions:
            circles_list.append([x, y_pos, 0.12])
    circles = np.array(circles_list[:N])
    
    # Strategy 2: Random initial with jitter
    circles_rand = np.zeros((N, 3))
    for i in range(N):
        circles_rand[i, 0] = 0.1 + 0.8 * np.random.random()
        circles_rand[i, 1] = 0.1 + 0.8 * np.random.random()
        circles_rand[i, 2] = 0.08
    
    strategies = [circles, circles_rand]
    
    for init_circles in strategies:
        x0 = init_circles[:, 0].copy()
        y0 = init_circles[:, 1].copy()
        r0 = init_circles[:, 2].copy()
        
        # Try multiple random perturbations
        for trial in range(3):
            x_try = x0 + 0.05 * np.random.randn(N) if trial > 0 else x0.copy()
            y_try = y0 + 0.05 * np.random.randn(N) if trial > 0 else y0.copy()
            r_try = r0.copy()
            
            params0 = np.concatenate([x_try, y_try, r_try])
            
            # Bounds
            bounds = []
            for i in range(N):
                bounds.append((0.01, 0.99))  # x
                bounds.append((0.01, 0.99))  # y
                bounds.append((0.01, 0.45))  # r
            
            # Optimize
            result = minimize(compute_objective, params0, method='L-BFGS-B', 
                            bounds=bounds, options={'maxiter': 200, 'ftol': 1e-10})
            
            if result.success:
                x_opt = result.x[:N]
                y_opt = result.x[N:2*N]
                r_opt = result.x[2*N:]
                
                # Clip to valid range
                x_opt = np.clip(x_opt, 0.01, 0.99)
                y_opt = np.clip(y_opt, 0.01, 0.99)
                r_opt = np.clip(r_opt, 0.01, 0.45)
                
                # Ensure circles fit in bounds
                for i in range(N):
                    r_opt[i] = min(r_opt[i], x_opt[i], 1 - x_opt[i], y_opt[i], 1 - y_opt[i])
                
                result_circles = np.column_stack([x_opt, y_opt, r_opt])
                
                if is_valid(result_circles):
                    score = np.sum(result_circles[:, 2])
                    if score > best_score:
                        best_score = score
                        best_result = result_circles
    
    # If no valid result found, use a safe grid
    if best_result is None or not is_valid(best_result):
        side = 6  # 6x6 grid, use 26 of 36 cells
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    x = (i + 0.5) / side
                    y = (j + 0.5) / side
                    r = 0.5 / side - 0.005
                    cells.append([x, y, r])
        return np.array(cells)
    
    return best_result
# EVOLVE-BLOCK-END
