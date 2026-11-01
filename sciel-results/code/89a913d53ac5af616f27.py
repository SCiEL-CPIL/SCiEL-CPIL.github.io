"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def unpack(x):
        w = x[0]
        h = 2.0 - w
        circles = x[1:].reshape(N, 3)
        return w, h, circles
    
    def objective(x):
        w, h, circles = unpack(x)
        return -np.sum(circles[:, 2])
    
    def constraints(x):
        w, h, circles = unpack(x)
        cons = []
        
        # Rectangle bounds
        cons.append({'type': 'ineq', 'fun': lambda x: x[0] - 0.1})
        cons.append({'type': 'ineq', 'fun': lambda x: 1.9 - x[0]})
        
        for i in range(N):
            idx_cx = 1 + 3*i
            idx_cy = 2 + 3*i
            idx_r = 3 + 3*i
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[1+3*i] - x[3+3*i]})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[0] - x[1+3*i] - x[3+3*i]})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2+3*i] - x[3+3*i]})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 2 - x[0] - x[2+3*i] - x[3+3*i]})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3+3*i]})
        
        for i in range(N):
            for j in range(i+1, N):
                cons.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: 
                    (x[1+3*i]-x[1+3*j])**2 + (x[2+3*i]-x[2+3*j])**2 - (x[3+3*i]+x[3+3*j])**2})
        
        return cons
    
    def bounds():
        b = [(0.1, 1.9)]
        for i in range(N):
            b.append((0, 2.0))
            b.append((0, 2.0))
            b.append((0, 1.0))
        return b
    
    def make_grid_layout(w, h):
        best_layout = None
        best_sum = 0
        
        for rows in range(3, 8):
            for cols in range(3, 8):
                if rows * cols < N:
                    continue
                positions = []
                for i in range(N):
                    row = i // cols
                    col = i % cols
                    x = (col + 0.5) * w / cols
                    y = (row + 0.5) * h / rows
                    positions.append([x, y])
                
                cell_w = w / cols
                cell_h = h / rows
                r = 0.4 * min(cell_w, cell_h)
                
                circles = np.array([[p[0], p[1], r] for p in positions])
                total_r = np.sum(circles[:, 2])
                
                if total_r > best_sum:
                    best_sum = total_r
                    best_layout = circles
        
        if best_layout is None:
            cols = 5
            positions = []
            for i in range(N):
                row = i // cols
                col = i % cols
                x = (col + 0.5) * w / cols
                y = (row + 0.5) * h / cols
                positions.append([x, y])
            r = 0.15
            best_layout = np.array([[p[0], p[1], r] for p in positions])
        
        return best_layout
    
    best_result = None
    best_obj = -float('inf')
    
    w_candidates = [0.8, 0.9, 1.0, 1.1, 1.2]
    
    for w_init in w_candidates:
        if time.time() > deadline:
            break
            
        h_init = 2.0 - w_init
        circles_init = make_grid_layout(w_init, h_init)
        
        x0 = np.concatenate([[w_init], circles_init.flatten()])
        
        try:
            res = minimize(
                objective, x0,
                method='SLSQP',
                bounds=bounds(),
                constraints=constraints,
                options={'maxiter': 300, 'ftol': 1e-9}
            )
            
            w_opt, h_opt, circles_opt = unpack(res.x)
            
            valid = True
            for i in range(N):
                cx, cy, r = circles_opt[i]
                if r < 1e-8 or cx < r - 1e-5 or w_opt - cx < r - 1e-5 or cy < r - 1e-5 or h_opt - cy < r - 1e-5:
                    valid = False
                    break
            
            if valid and not np.any(np.isnan(circles_opt)):
                obj_val = np.sum(circles_opt[:, 2])
                if obj_val > best_obj:
                    best_obj = obj_val
                    best_result = (w_opt, h_opt, circles_opt)
        except Exception:
            continue
    
    if best_result is None:
        w_opt, h_opt = 1.0, 1.0
        circles_opt = make_grid_layout(w_opt, h_opt)
    
    result = np.column_stack([circles_opt[:, 0], circles_opt[:, 1], circles_opt[:, 2]])
    
    return result

# EVOLVE-BLOCK-END
