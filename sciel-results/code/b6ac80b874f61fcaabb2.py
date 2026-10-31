"""Rewritten program: circle packing in perimeter-4 rectangle using SLSQP."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    best_sum_r = 0.0
    best_config = None
    best_W = 1.0
    best_H = 1.0
    
    # Try several aspect ratios
    aspect_ratios = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    
    for ratio in aspect_ratios:
        if time.time() > deadline - 30:
            break
        
        # W + H = 2, W/H = ratio
        # W = ratio * H, so (ratio + 1) * H = 2, H = 2/(ratio+1)
        H = 2.0 / (ratio + 1.0)
        W = ratio * H
        
        # Initial positions: hexagonal-like grid
        cols = int(np.ceil(np.sqrt(N * W / H)))
        if cols < 1: cols = 1
        rows = int(np.ceil(N / cols))
        
        xs = np.zeros(N)
        ys = np.zeros(N)
        for i in range(N):
            c = i % cols
            r_idx = i // cols
            # Hexagonal offset
            x_off = (c + 0.5) * W / cols
            y_off = (r_idx + 0.5) * H / rows
            if r_idx % 2 == 1 and c < cols - 1:
                x_off += 0.25 * W / cols
            xs[i] = x_off
            ys[i] = y_off
        
        # Initial radii: small
        r_max_est = np.sqrt(0.9 * W * H / (N * np.pi))
        r_init = 0.3 * r_max_est
        
        # We'll optimize over positions and radii
        # Variables: x_0, y_0, r_0, x_1, y_1, r_1, ...
        # Total: 3*N variables
        
        def objective(params):
            xs_opt = params[0::3]
            ys_opt = params[1::3]
            rs_opt = params[2::3]
            # Maximize sum(r) => minimize -sum(r)
            return -np.sum(rs_opt)
        
        def constraints_func(params):
            xs_opt = params[0::3]
            ys_opt = params[1::3]
            rs_opt = params[2::3]
            
            # Wall constraints: x_i - r_i >= 0, W - x_i - r_i >= 0, etc.
            cons = []
            for i in range(N):
                # Left
                cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i] - p[3*i+2]})
                # Right
                cons.append({'type': 'ineq', 'fun': lambda p, i=i: W - p[3*i] - p[3*i+2]})
                # Bottom
                cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+1] - p[3*i+2]})
                # Top
                cons.append({'type': 'ineq', 'fun': lambda p, i=i: H - p[3*i+1] - p[3*i+2]})
            
            # Pairwise: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
            for i in range(N):
                for j in range(i+1, N):
                    cons.append({
                        'type': 'ineq',
                        'fun': lambda p, i=i, j=j: (p[3*i]-p[3*j])**2 + (p[3*i+1]-p[3*j+1])**2 - (p[3*i+2]+p[3*j+2])**2
                    })
            
            return cons
        
        # Build initial params
        params0 = np.zeros(3*N)
        params0[0::3] = xs
        params0[1::3] = ys
        params0[2::3] = r_init
        
        # Try SLSQP
        try:
            result = minimize(
                objective,
                params0,
                method='SLSQP',
                constraints=constraints_func(params0),
                bounds=[(0, W), (0, H), (0.001, r_max_est*1.5)] * N,
                options={'maxiter': 500, 'ftol': 1e-8, 'disp': False}
            )
            
            if result.success or result.fun < 0:  # result.fun is -sum(r)
                # Extract solution
                xs_sol = result.x[0::3]
                ys_sol = result.x[1::3]
                rs_sol = result.x[2::3]
                
                # Verify validity
                valid = True
                for i in range(N):
                    if xs_sol[i] < -1e-8 or xs_sol[i] > W + 1e-8:
                        valid = False
                        break
                    if ys_sol[i] < -1e-8 or ys_sol[i] > H + 1e-8:
                        valid = False
                        break
                    if xs_sol[i] < rs_sol[i] - 1e-8 or xs_sol[i] > W - rs_sol[i] + 1e-8:
                        valid = False
                        break
                    if ys_sol[i] < rs_sol[i] - 1e-8 or ys_sol[i] > H - rs_sol[i] + 1e-8:
                        valid = False
                        break
                for i in range(N):
                    if not valid:
                        break
                    for j in range(i+1, N):
                        dx = xs_sol[i] - xs_sol[j]
                        dy = ys_sol[i] - ys_sol[j]
                        dist_sq = dx*dx + dy*dy
                        min_dist = rs_sol[i] + rs_sol[j]
                        if dist_sq < min_dist*min_dist - 1e-8:
                            valid = False
                            break
                
                if valid:
                    sum_r = np.sum(rs_sol)
                    if sum_r > best_sum_r:
                        best_sum_r = sum_r
                        best_config = np.column_stack((xs_sol, ys_sol, rs_sol))
                        best_W = W
                        best_H = H
                        
        except Exception:
            pass
        
    # Fallback: simple grid
    if best_config is None:
        W = 1.0
        H = 1.0
        cols = 5
        rows = 5
        cells = []
        for i in range(N):
            c = i % cols
            r_idx = i // cols
            x = (c + 0.5) * W / cols
            y = (r_idx + 0.5) * H / rows
            r = 0.5 * min(W, H) / max(cols, rows) - 1e-3
            cells.append([x, y, r])
        best_config = np.array(cells)
    
    return best_config

# EVOLVE-BLOCK-END
