import numpy as np
from scipy.optimize import minimize
import time

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    best_sum = 0
    best_sol = None
    
    # Objective function: Minimize -sum(r) + penalty
    def objective(params):
        # params: [x0, y0, r0, x1, y1, r1, ...]
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        
        # Penalty for negative radii
        pen_r = np.sum(np.maximum(0, -r)**2)
        
        # Penalty for boundary violations
        # dist to walls
        d_w = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
        pen_w = np.sum(np.maximum(0, r - d_w)**2)
        
        # Penalty for overlaps
        pen_o = 0
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                viol = r[i] + r[j] - d
                if viol > 0:
                    pen_o += viol**2
                    
        return -np.sum(r) + 100 * (pen_r + pen_w + pen_o)

    def run_optimization(x0, y0, r0):
        params0 = np.concatenate([x0, y0, r0])
        # Use Powell as it is derivative-free and robust
        res = minimize(objective, params0, method='Powell', options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
        return res.x, res.fun

    # Generate initial configurations
    inits = []
    
    # 1. Hexagonal packing
    # Approximate hex grid
    # We want to fit 26 circles.
    # Let's try to place them in a hex pattern and scale.
    # Simple hex generation
    pts = []
    # Rows
    # 5 rows of 5? No, hex is staggered.
    # Let's just use a random initialization and a grid initialization.
    
    # Grid init (similar to seed but slightly better spaced if possible)
    side = int(np.ceil(np.sqrt(N)))
    for i in range(side):
        for j in range(side):
            if len(pts) < N:
                pts.append([(i+0.5)/side, (j+0.5)/side])
    if len(pts) < N:
        # Fill remaining randomly
        while len(pts) < N:
            pts.append([np.random.rand(), np.random.rand()])
            
    x0_grid = np.array([p[0] for p in pts])
    y0_grid = np.array([p[1] for p in pts])
    r0_grid = np.full(N, 0.05) # Small initial radius
    
    inits.append((x0_grid, y0_grid, r0_grid))
    
    # 2. Random inits
    for _ in range(20):
        x0_r = np.random.rand(N)
        y0_r = np.random.rand(N)
        r0_r = np.full(N, 0.05)
        inits.append((x0_r, y0_r, r0_r))
        
    # 3. "Clustered" init: Group circles to encourage packing
    # Just random is usually fine with Powell
    
    for idx, (x0, y0, r0) in enumerate(inits):
        if time.time() > deadline:
            break
            
        try:
            sol, fun = run_optimization(x0, y0, r0)
            # Extract solution
            x_sol = sol[0::3]
            y_sol = sol[1::3]
            r_sol = sol[2::3]
            
            # Check validity
            # Radii must be positive
            if np.any(r_sol < 0):
                continue
                
            # Check constraints strictly
            valid = True
            for i in range(N):
                if x_sol[i] < -1e-6 or x_sol[i] > 1+1e-6 or y_sol[i] < -1e-6 or y_sol[i] > 1+1e-6:
                    valid = False
                    break
                if r_sol[i] > x_sol[i] + 1e-6 or r_sol[i] > 1-x_sol[i] + 1e-6 or r_sol[i] > y_sol[i] + 1e-6 or r_sol[i] > 1-y_sol[i] + 1e-6:
                    valid = False
                    break
            if not valid:
                continue
                
            for i in range(N):
                for j in range(i+1, N):
                    d = np.sqrt((x_sol[i]-x_sol[j])**2 + (y_sol[i]-y_sol[j])**2)
                    if d < r_sol[i] + r_sol[j] - 1e-6:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                # Try to fix by shrinking radii? Or just skip.
                # If penalty was high, it might be invalid.
                # Let's try to project onto feasible set by reducing radii.
                # Simple projection: reduce all radii by max violation.
                max_viol = 0
                for i in range(N):
                    d_w = np.min([x_sol[i], 1-x_sol[i], y_sol[i], 1-y_sol[i]])
                    max_viol = max(max_viol, r_sol[i] - d_w)
                for i in range(N):
                    for j in range(i+1, N):
                        d = np.sqrt((x_sol[i]-x_sol[j])**2 + (y_sol[i]-y_sol[j])**2)
                        max_viol = max(max_viol, (r_sol[i] + r_sol[j] - d)/2)
                        
                if max_viol > 0:
                    r_sol = r_sol - max_viol
                    if np.any(r_sol < 0):
                        continue
                    # Re-check
                    valid = True
                    for i in range(N):
                        d_w = np.min([x_sol[i], 1-x_sol[i], y_sol[i], 1-y_sol[i]])
                        if r_sol[i] > d_w + 1e-6:
                            valid = False
                            break
                    if not valid: continue
                    for i in range(N):
                        for j in range(i+1, N):
                            d = np.sqrt((x_sol[i]-x_sol[j])**2 + (y_sol[i]-y_sol[j])**2)
                            if d < r_sol[i] + r_sol[j] - 1e-6:
                                valid = False
                                break
                        if not valid: break
                    if not valid: continue

            sum_r = np.sum(r_sol)
            if sum_r > best_sum:
                best_sum = sum_r
                best_sol = np.column_stack([x_sol, y_sol, r_sol])
                
        except Exception:
            continue
            
    if best_sol is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_sol
