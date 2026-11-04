"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    best_value = 0.0
    best_solution = None

    # Helper to check validity and value
    def evaluate_solution(v):
        w = v[0]
        h = 2.0 - w
        x = v[1:N+1]
        y = v[N+1:2*N+1]
        r = v[2*N+1:3*N+1]
        
        # Check bounds
        if np.any(r < 1e-6) or w <= 0 or h <= 0:
            return 0.0, False
        if np.any(x < r - 1e-5) or np.any(x > w - r + 1e-5) or \
           np.any(y < r - 1e-5) or np.any(y > h - r + 1e-5):
            return 0.0, False
            
        # Check overlaps
        diff_x = x[:, np.newaxis] - x[np.newaxis, :]
        diff_y = y[:, np.newaxis] - y[np.newaxis, :]
        dist = np.sqrt(diff_x**2 + diff_y**2)
        np.fill_diagonal(dist, np.inf)
        
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        if np.any(r_sum > dist + 1e-5):
            return 0.0, False
            
        val = np.sum(r)
        return val, True

    # Hexagonal initialization
    def init_hex(w, h):
        # Create a hexagonal grid of points
        # We want 21 points.
        # Let's try rows of 4, 5, 4, 5, 3
        rows = [4, 5, 4, 5, 3]
        points = []
        r_init = 0.1
        # Center the grid in the rectangle
        # Approximate width of grid: max_row * 2 * r_init
        # Approximate height: len(rows) * sqrt(3) * r_init
        
        # Scale r_init to fit
        max_row = max(rows)
        grid_w = max_row * 2.0 * r_init
        grid_h = len(rows) * np.sqrt(3) * r_init
        
        # Adjust r_init so grid fits in w, h with some margin
        r_init = min(w / grid_w, h / grid_h) * 0.8 * r_init
        if r_init < 0.01:
            r_init = 0.01
            
        y_offset = (h - grid_h) / 2.0
        y_curr = y_offset
        
        for count in rows:
            x_offset = (w - count * 2.0 * r_init) / 2.0
            for i in range(count):
                x = x_offset + i * 2.0 * r_init
                y = y_curr
                points.append([x, y, r_init])
            y_curr += np.sqrt(3) * r_init
            
        # Pad if fewer than N
        while len(points) < N:
            points.append([w/2, h/2, r_init])
            
        points = points[:N]
        
        x = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])
        r = np.array([p[2] for p in points])
        
        return np.concatenate(([w], x, y, r))

    # Objective function
    def objective(v):
        w = v[0]
        h = 2.0 - w
        x = v[1:N+1]
        y = v[N+1:2*N+1]
        r = v[2*N+1:3*N+1]
        
        # Bounds penalty
        pen_x_low = np.maximum(0, r - x)
        pen_x_high = np.maximum(0, x + r - w)
        pen_y_low = np.maximum(0, r - y)
        pen_y_high = np.maximum(0, y + r - h)
        
        bound_pen = np.sum(pen_x_low**2) + np.sum(pen_x_high**2) + \
                    np.sum(pen_y_low**2) + np.sum(pen_y_high**2)
        
        # Overlap penalty
        diff_x = x[:, np.newaxis] - x[np.newaxis, :]
        diff_y = y[:, np.newaxis] - y[np.newaxis, :]
        dist = np.sqrt(diff_x**2 + diff_y**2)
        np.fill_diagonal(dist, np.inf)
        
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        violation = np.maximum(0, r_sum - dist)
        
        tri_idx = np.triu_indices(N, k=1)
        overlap_pen = np.sum(violation[tri_idx]**2)
        
        total_pen = bound_pen + overlap_pen
        
        sum_r = np.sum(r)
        
        # Adaptive penalty weight? 
        # Start with high weight to enforce constraints, then lower to maximize?
        # For simplicity, use a fixed high weight.
        weight = 100.0
        
        return -sum_r + weight * total_pen

    bounds = [(0.1, 1.9)] + [(None, None)] * (3*N - 1)
    # We can set lower bound for r to be small positive
    # But L-BFGS-B bounds are per variable.
    # v[0] is w. v[1..N] x. v[N+1..2N] y. v[2N+1..3N] r.
    bounds_list = [(0.1, 1.9)]
    for i in range(N):
        bounds_list.append((None, None)) # x
    for i in range(N):
        bounds_list.append((None, None)) # y
    for i in range(N):
        bounds_list.append((1e-6, None)) # r

    # Try multiple initializations
    w_starts = [0.8, 0.9, 1.0, 1.1, 1.2, 1.4]
    
    for w_start in w_starts:
        if time.time() > deadline:
            break
            
        x0 = init_hex(w_start, 2.0 - w_start)
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds_list, 
                           options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.success or res.fun < 0: # res.fun < 0 means sum_r > penalty
                val, valid = evaluate_solution(res.x)
                if valid and val > best_value:
                    best_value = val
                    best_solution = res.x
                    
        except Exception as e:
            pass

    # If we found a solution, try to polish it
    if best_solution is not None and time.time() < deadline - 10:
        try:
            # Polish with tighter tolerances and higher penalty weight to ensure validity
            def objective_polish(v):
                w = v[0]
                h = 2.0 - w
                x = v[1:N+1]
                y = v[N+1:2*N+1]
                r = v[2*N+1:3*N+1]
                
                pen_x_low = np.maximum(0, r - x)
                pen_x_high = np.maximum(0, x + r - w)
                pen_y_low = np.maximum(0, r - y)
                pen_y_high = np.maximum(0, y + r - h)
                
                bound_pen = np.sum(pen_x_low**2) + np.sum(pen_x_high**2) + \
                            np.sum(pen_y_low**2) + np.sum(pen_y_high**2)
                
                diff_x = x[:, np.newaxis] - x[np.newaxis, :]
                diff_y = y[:, np.newaxis] - y[np.newaxis, :]
                dist = np.sqrt(diff_x**2 + diff_y**2)
                np.fill_diagonal(dist, np.inf)
                
                r_sum = r[:, np.newaxis] + r[np.newaxis, :]
                violation = np.maximum(0, r_sum - dist)
                
                tri_idx = np.triu_indices(N, k=1)
                overlap_pen = np.sum(violation[tri_idx]**2)
                
                total_pen = bound_pen + overlap_pen
                sum_r = np.sum(r)
                
                weight = 1000.0
                return -sum_r + weight * total_pen

            res_polish = minimize(objective_polish, best_solution, method='L-BFGS-B', 
                                  bounds=bounds_list, options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
            
            val, valid = evaluate_solution(res_polish.x)
            if valid and val > best_value:
                best_value = val
                best_solution = res_polish.x
                
        except Exception:
            pass

    if best_solution is None:
        # Fallback to seed-like behavior if optimization fails
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    # Format output: (n, 3) array of [x, y, r]
    w = best_solution[0]
    h = 2.0 - w
    x = best_solution[1:N+1]
    y = best_solution[N+1:2*N+1]
    r = best_solution[2*N+1:3*N+1]
    
    result = np.column_stack((x, y, r))
    return result

# EVOLVE-BLOCK-END
