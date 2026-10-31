"""
Circle packing in perimeter-4 rectangle.
Strategy: Constrained Non-Linear Programming (NLP).
We optimize the positions (x, y), radii (r), and width (w) of the rectangle.
Height h is determined by h = 2 - w.
Constraints:
1. Non-overlap: dist(i, j) >= r_i + r_j
2. Boundary: 0 <= x_i - r_i, x_i + r_i <= w, 0 <= y_i - r_i, y_i + r_i <= h
Objective: Maximize sum(r_i)
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    best_value = 0.0
    best_pos = None

    def check_and_value(pos):
        """
        pos: (N, 3) array of [x, y, r]
        Returns (value, is_valid)
        """
        x = pos[:, 0]
        y = pos[:, 1]
        r = pos[:, 2]
        
        if np.any(r <= 0):
            return 0.0, False
            
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                min_dist = r[i] + r[j]
                if dist_sq < min_dist*min_dist - 1e-9:
                    return 0.0, False
        
        # Compute bounding box
        min_x = np.min(x - r)
        max_x = np.max(x + r)
        min_y = np.min(y - r)
        max_y = np.max(y + r)
        
        w = max_x - min_x
        h = max_y - min_y
        
        if 2 * (w + h) > 4.0 + 1e-9:
            return 0.0, False
            
        return np.sum(r), True

    def solve_for_aspect(w0):
        """
        Solve the packing problem for a fixed aspect ratio w0.
        h0 = 2 - w0.
        Variables: x, y, r (3N variables). w is fixed.
        We use SLSQP for the constrained optimization.
        """
        h0 = 2.0 - w0
        if h0 <= 0.1: return None
        
        # Initial guess: Grid
        side = int(np.ceil(np.sqrt(N)))
        # Adjust grid to fit N better if possible, e.g. 5x5 grid for 21 circles
        # Let's try a few grid layouts
        # For 21 circles, 5x5 is 25 cells. 4x5 is 20. 
        # Let's just use a uniform grid of side x side and take the first N.
        
        # Better init: place circles in a grid that fits in w0 x h0
        # Estimate max radius
        max_r_est = min(w0/side, h0/side) * 0.9
        
        x0 = np.zeros(N)
        y0 = np.zeros(N)
        r0 = np.zeros(N)
        
        # Map N circles to a grid
        # If N=21, use 5x5 grid
        cols = 5
        rows = 5
        for i in range(N):
            row, col = divmod(i, cols)
            x0[i] = (col + 0.5) / cols * w0
            y0[i] = (row + 0.5) / rows * h0
            r0[i] = max_r_est * 0.8 # Start slightly smaller to allow growth
            
        # Ensure initial state is feasible (no overlaps)
        # If not, shrink radii
        # Simple check
        for i in range(N):
            for j in range(i+1, N):
                dx = x0[i] - x0[j]
                dy = y0[i] - y0[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r0[i] + r0[j]:
                    # Shrink both
                    scale = dist / (r0[i] + r0[j]) * 0.95
                    r0[i] *= scale
                    r0[j] *= scale

        x0 = np.clip(x0, r0, w0 - r0)
        y0 = np.clip(y0, r0, h0 - r0)
        
        vars0 = np.concatenate([x0, y0, r0])
        
        # Objective: Minimize -sum(r)
        def objective(vars):
            x = vars[:N]
            y = vars[N:2*N]
            r = vars[2*N:]
            return -np.sum(r)
            
        # Constraints
        # 1. Non-overlap: dist(i,j) - r_i - r_j >= 0
        # 2. Boundaries: x_i - r_i >= 0, w - (x_i + r_i) >= 0, etc.
        
        # SLSQP requires equality and inequality constraints in the form g(vars) >= 0
        # We will define them explicitly.
        
        cons = []
        
        # Boundary constraints
        # x_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2*N+i]})
        # w - (x_i + r_i) >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: w0 - (v[i] + v[2*N+i])})
        # y_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[N+i] - v[2*N+i]})
        # h - (y_i + r_i) >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: h0 - (v[N+i] + v[2*N+i])})
        
        # Non-overlap constraints
        for i in range(N):
            for j in range(i+1, N):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda v, i=i, j=j: np.sqrt((v[i]-v[j])**2 + (v[N+i]-v[N+j])**2) - v[2*N+i] - v[2*N+j]
                })
                
        # Bounds
        # x in [0, w0], y in [0, h0], r in [0, min(w0,h0)/2]
        max_r = min(w0, h0) * 0.5
        bounds = []
        for i in range(N):
            bounds.append((0, w0)) # x
            bounds.append((0, h0)) # y
            bounds.append((0.01, max_r)) # r
            
        try:
            # SLSQP is good for small problems with constraints
            res = minimize(objective, vars0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 200, 'ftol': 1e-8})
            
            if res.success or res.fun < -best_value:
                x_res = res.x[:N]
                y_res = res.x[N:2*N]
                r_res = res.x[2*N:]
                
                # Refine with L-BFGS-B using penalty to ensure strict feasibility and potentially better packing
                # L-BFGS-B doesn't handle non-linear constraints well, but we can use a penalty method
                # Or just return the SLSQP result if it's good.
                # Let's try a short L-BFGS-B refinement with a high penalty for violations.
                
                def penalty_objective(vars):
                    x = vars[:N]
                    y = vars[N:2*N]
                    r = vars[2*N:]
                    pen = 0.0
                    # Boundary penalties
                    pen += np.sum(np.maximum(0, r - x)**2)
                    pen += np.sum(np.maximum(0, r - y)**2)
                    pen += np.sum(np.maximum(0, x + r - w0)**2)
                    pen += np.sum(np.maximum(0, y + r - h0)**2)
                    
                    # Overlap penalties
                    for i in range(N):
                        for j in range(i+1, N):
                            dx = x[i] - x[j]
                            dy = y[i] - y[j]
                            dist = np.sqrt(dx*dx + dy*dy)
                            min_d = r[i] + r[j]
                            if dist < min_d:
                                pen += (min_d - dist)**2
                                
                    return -np.sum(r) + 100.0 * pen
                    
                # Refine from SLSQP solution
                # Ensure start is feasible for L-BFGS-B (it handles bounds, not constraints)
                # We project to bounds
                x_ref = np.clip(x_res, 0, w0)
                y_ref = np.clip(y_res, 0, h0)
                r_ref = np.clip(r_res, 0.01, max_r)
                
                # Check if SLSQP solution is already very good
                val_check, valid_check = check_and_value(np.column_stack((x_res, y_res, r_res)))
                
                if valid_check:
                     # Try to improve with L-BFGS-B
                     p_ref = np.concatenate([x_ref, y_ref, r_ref])
                     try:
                         res_ref = minimize(penalty_objective, p_ref, method='L-BFGS-B', bounds=bounds,
                                            options={'maxiter': 50, 'ftol': 1e-9})
                         x_ref2 = res_ref.x[:N]
                         y_ref2 = res_ref.x[N:2*N]
                         r_ref2 = res_ref.x[2*N:]
                         
                         val_ref, valid_ref = check_and_value(np.column_stack((x_ref2, y_ref2, r_ref2)))
                         if valid_ref and val_ref > val_check:
                             return val_ref, np.column_stack((x_ref2, y_ref2, r_ref2))
                     except:
                         pass
                     return val_check, np.column_stack((x_res, y_res, r_res))
                else:
                     # SLSQP failed to find feasible solution, try L-BFGS-B from scratch?
                     # Or just return None
                     return None, None
                     
        except Exception:
            return None, None

    # Multi-start over aspect ratios
    # w + h = 2. w in [0.5, 1.5]
    aspects = np.linspace(0.5, 1.5, 10)
    
    for w0 in aspects:
        if time.time() > deadline - 5:
            break
            
        val, pos = solve_for_aspect(w0)
        if val is not None and val > best_value:
            best_value = val
            best_pos = pos
            
    # Fallback: Simple grid in 1x1 box
    if best_pos is None:
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_pos = np.array(cells[:N])
        # Verify fallback
        val, valid = check_and_value(best_pos)
        if valid:
            best_value = val
        else:
            # Even simpler fallback: 1x1 grid with smaller radius
            side = 5
            cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side * 0.9) for i in range(side) for j in range(side)]
            best_pos = np.array(cells[:N])
            val, valid = check_and_value(best_pos)
            if valid:
                best_value = val
            else:
                # Should not happen
                best_pos = np.zeros((N, 3))
                best_pos[:, 0] = 0.5
                best_pos[:, 1] = 0.5
                best_pos[:, 2] = 0.01

    return best_pos[:N]
