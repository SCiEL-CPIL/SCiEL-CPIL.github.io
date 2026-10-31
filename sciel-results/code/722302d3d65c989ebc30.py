"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21
import numpy as np
import time
import random
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    best_sum_r = 0.0
    best_result = None
    
    def validate_and_score(xs, ys, rs, w):
        h = 2.0 - w
        if w <= 0 or h <= 0:
            return -1.0
        # Check wall constraints
        for i in range(N):
            if xs[i] < rs[i] - 1e-9: return -1.0
            if xs[i] > w - rs[i] + 1e-9: return -1.0
            if ys[i] < rs[i] - 1e-9: return -1.0
            if ys[i] > h - rs[i] + 1e-9: return -1.0
        # Check circle constraints
        for i in range(N):
            for j in range(i+1, N):
                dist_sq = (xs[i]-xs[j])**2 + (ys[i]-ys[j])**2
                min_dist = rs[i] + rs[j]
                if dist_sq < min_dist**2 - 1e-9:
                    return -1.0
        return np.sum(rs)

    def solve_penalty(w_val, init_xs, init_ys, init_rs, time_limit):
        h_val = 2.0 - w_val
        
        def obj_pen(v):
            xs = v[0:3*N:3]
            ys = v[1:3*N:3]
            rs = v[2:3*N:3]
            
            penalty = 0.0
            # Walls
            for i in range(N):
                if xs[i] < rs[i]: penalty += (rs[i] - xs[i])**2
                if xs[i] > w_val - rs[i]: penalty += (xs[i] - (w_val - rs[i]))**2
                if ys[i] < rs[i]: penalty += (rs[i] - ys[i])**2
                if ys[i] > h_val - rs[i]: penalty += (ys[i] - (h_val - rs[i]))**2
                
            # Circles
            for i in range(N):
                for j in range(i+1, N):
                    dist_sq = (xs[i]-xs[j])**2 + (ys[i]-ys[j])**2
                    min_dist = rs[i] + rs[j]
                    if dist_sq < min_dist**2:
                        diff = min_dist**2 - dist_sq
                        penalty += diff**2
            
            return -np.sum(rs) + 1000.0 * penalty
            
        v0 = np.concatenate([init_xs, init_ys, init_rs])
        
        # Use Nelder-Mead for robustness with non-smooth penalties initially
        # Then refine with BFGS if possible? Nelder-Mead is often safer for these.
        # Let's try L-BFGS-B if we can compute gradients, but numerical gradients are expensive.
        # Nelder-Mead with high maxiter is a good bet.
        
        res = minimize(obj_pen, v0, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
        
        # Check if result is valid
        xs = res.x[0:3*N:3]
        ys = res.x[1:3*N:3]
        rs = res.x[2:3*N:3]
        
        score = validate_and_score(xs, ys, rs, w_val)
        
        # If invalid, try to "repair" or just reject? 
        # If penalty is low but validation fails due to tiny epsilon, we might try to project.
        # For now, if valid, return it.
        if score > 0:
            return score, res.x
        
        # If invalid but penalty is very low, it might be a near-miss.
        # We could try a local search to fix it, but let's keep it simple first.
        return -1.0, None

    # Search over aspect ratios
    # w + h = 2. Symmetry w<=h => w<=1.
    # But we can search w in [0.5, 1.5] to be safe, though [0.5, 1.0] covers unique shapes.
    
    w_values = np.linspace(0.6, 1.0, 15)
    
    # Also try some random w values
    for _ in range(10):
        w_vals_rand = np.random.uniform(0.5, 1.0)
        w_values = np.append(w_values, w_vals_rand)
    
    w_values = np.unique(np.round(w_values, 3))
    
    # Sort by likely density? Square (w=1) is often good.
    # Let's just iterate.
    
    for w in w_values:
        if time.time() - start_time > deadline:
            break
            
        h = 2.0 - w
        
        # Generate multiple initial configurations
        # 1. Grid based
        # 2. Hexagonal based
        
        n_configs = 5
        
        for config_idx in range(n_configs):
            if time.time() - start_time > deadline:
                break
                
            xs = np.zeros(N)
            ys = np.zeros(N)
            rs = np.zeros(N)
            
            if config_idx == 0:
                # Simple grid
                # Estimate cols/rows
                # Area = w*h. N circles.
                # r ~ sqrt(w*h / (N*pi))
                # Let's try a few grid splits
                best_grid_score = -1.0
                best_grid_v = None
                
                # Try splits that sum to >= N
                # 4x6, 5x5, 6x4, 3x7, 7x3
                splits = [(4, 6), (5, 5), (6, 4), (3, 7), (7, 3), (5, 6), (6, 5)]
                
                for rows, cols in splits:
                    if rows * cols < N:
                        continue
                        
                    xs_grid = []
                    ys_grid = []
                    for i in range(rows):
                        for j in range(cols):
                            if len(xs_grid) >= N:
                                break
                            x = (j + 0.5) * w / cols
                            y = (i + 0.5) * h / rows
                            xs_grid.append(x)
                            ys_grid.append(y)
                    
                    xs_grid = np.array(xs_grid)
                    ys_grid = np.array(ys_grid)
                    
                    # Calculate min distance for initial r
                    min_dist = np.inf
                    for i in range(N):
                        for j in range(i+1, N):
                            d = np.sqrt((xs_grid[i]-xs_grid[j])**2 + (ys_grid[i]-ys_grid[j])**2)
                            min_dist = min(min_dist, d)
                    for i in range(N):
                        d_w = min(xs_grid[i], w-xs_grid[i], ys_grid[i], h-ys_grid[i])
                        min_dist = min(min_dist, d_w)
                    
                    r_init = min_dist / 2.0 * 0.8
                    rs_init = np.full(N, r_init)
                    
                    score, res_v = solve_penalty(w, xs_grid, ys_grid, rs_init, 5.0)
                    
                    if score > best_grid_score:
                        best_grid_score = score
                        best_grid_v = res_v
                        
                if best_grid_v is not None:
                    xs = best_grid_v[0:3*N:3]
                    ys = best_grid_v[1:3*N:3]
                    rs = best_grid_v[2:3*N:3]
                    if best_grid_score > best_sum_r:
                        best_sum_r = best_grid_score
                        best_result = np.column_stack([xs, ys, rs])
                        
            elif config_idx == 1:
                # Hexagonal
                # Estimate r
                # In hexagonal packing, density is pi / (2*sqrt(3)) ~ 0.9069
                # N * pi * r^2 ~ 0.9 * w * h
                # r ~ sqrt(0.9 * w * h / (N * pi))
                r_est = np.sqrt(0.9 * w * h / (N * np.pi))
                
                # Grid spacing
                dx = 2.0 * r_est
                dy = np.sqrt(3) * r_est
                
                # Number of cols
                n_cols = int(np.ceil(w / dx))
                n_rows = int(np.ceil(h / dy))
                
                # Adjust to fit N
                # This is heuristic. Let's just generate points and take first N.
                pts = []
                for row in range(n_rows + 2):
                    for col in range(n_cols + 2):
                        x = (col + 0.5) * dx
                        y = (row + 0.5) * dy
                        if row % 2 == 1:
                            x += 0.5 * dx
                        if x >= r_est and x <= w - r_est and y >= r_est and y <= h - r_est:
                            pts.append((x, y))
                
                if len(pts) >= N:
                    # Shuffle to pick random N? Or take first N?
                    # Random selection might find better local minima
                    random.shuffle(pts)
                    pts = pts[:N]
                    
                    xs = np.array([p[0] for p in pts])
                    ys = np.array([p[1] for p in pts])
                    
                    # Calculate min dist
                    min_dist = np.inf
                    for i in range(N):
                        for j in range(i+1, N):
                            d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                            min_dist = min(min_dist, d)
                    for i in range(N):
                        d_w = min(xs[i], w-xs[i], ys[i], h-ys[i])
                        min_dist = min(min_dist, d_w)
                    
                    r_init = min_dist / 2.0 * 0.8
                    rs = np.full(N, r_init)
                    
                    score, res_v = solve_penalty(w, xs, ys, rs, 5.0)
                    
                    if score > best_sum_r:
                        best_sum_r = score
                        best_result = np.column_stack([res_v[0:3*N:3], res_v[1:3*N:3], res_v[2:3*N:3]])
                        
            else:
                # Random jittered grid
                # Start with a rough grid and add jitter
                rows = 4
                cols = 6
                if rows * cols < N:
                    rows = 5
                    cols = 5
                    
                xs = []
                ys = []
                for i in range(rows):
                    for j in range(cols):
                        if len(xs) >= N: break
                        x = (j + 0.5) * w / cols
                        y = (i + 0.5) * h / rows
                        # Jitter
                        x += np.random.uniform(-0.1, 0.1) * (w/cols)
                        y += np.random.uniform(-0.1, 0.1) * (h/rows)
                        # Clip
                        x = np.clip(x, 0.05, w-0.05)
                        y = np.clip(y, 0.05, h-0.05)
                        xs.append(x)
                        ys.append(y)
                        
                xs = np.array(xs)
                ys = np.array(ys)
                
                min_dist = np.inf
                for i in range(N):
                    for j in range(i+1, N):
                        d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                        min_dist = min(min_dist, d)
                for i in range(N):
                    d_w = min(xs[i], w-xs[i], ys[i], h-ys[i])
                    min_dist = min(min_dist, d_w)
                
                r_init = min_dist / 2.0 * 0.5
                rs = np.full(N, r_init)
                
                score, res_v = solve_penalty(w, xs, ys, rs, 5.0)
                
                if score > best_sum_r:
                    best_sum_r = score
                    best_result = np.column_stack([res_v[0:3*N:3], res_v[1:3*N:3], res_v[2:3*N:3]])
        
    if best_result is None:
        # Fallback to simple grid in square
        w = 1.0
        h = 1.0
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3))
        return np.array(cells)
        
    return best_result

# EVOLVE-BLOCK-END
