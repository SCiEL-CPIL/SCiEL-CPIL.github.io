import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 21

def construct():
    # Objective function for Differential Evolution
    # We minimize: -sum(r) + penalty
    # Variables: [w, x_1..x_N, y_1..y_N, r_1..r_N]
    
    def penalty_objective(v):
        w = v[0]
        h = 2.0 - w
        
        # If w or h is non-positive, huge penalty
        if w <= 0.01 or h <= 0.01:
            return 1e10
            
        x = v[1:1+N]
        y = v[1+N:1+2*N]
        r = v[1+2*N:1+3*N]
        
        # Base objective: maximize sum of radii
        score = -np.sum(r)
        
        # Penalty for violations
        penalty = 0.0
        
        # 1. Circles inside box
        for i in range(N):
            # x_i >= r_i
            if x[i] < r[i]:
                penalty += (r[i] - x[i]) * 100.0
            # x_i + r_i <= w
            if x[i] + r[i] > w:
                penalty += (x[i] + r[i] - w) * 100.0
            # y_i >= r_i
            if y[i] < r[i]:
                penalty += (r[i] - y[i]) * 100.0
            # y_i + r_i <= h
            if y[i] + r[i] > h:
                penalty += (y[i] + r[i] - h) * 100.0
            # r_i >= 0
            if r[i] < 0:
                penalty += (-r[i]) * 100.0
                
        # 2. Non-overlap
        for i in range(N):
            for j in range(i+1, N):
                dist2 = (x[i]-x[j])**2 + (y[i]-y[j])**2
                sum_r = r[i] + r[j]
                if sum_r > 0:
                    min_dist2 = sum_r**2
                    if dist2 < min_dist2:
                        # Penalty proportional to overlap depth
                        overlap = sum_r - np.sqrt(dist2)
                        penalty += overlap * 100.0
                        
        return score + penalty

    # Bounds for Differential Evolution
    # w in [0.1, 1.9]
    # x, y in [0.0, 2.0] (loose, penalty handles it)
    # r in [0.0, 1.0] (loose, penalty handles it)
    
    bounds = [(0.1, 1.9)] + [(0.0, 2.0)] * (2*N) + [(0.0, 1.0)] * N
    
    # Initial strategy: Use a grid-like initialization for DE to help it
    # DE generates random points, but we can seed it or just let it run.
    # For N=21, the search space is 64D. DE is slow but robust.
    # We need to be careful with time. 120s limit.
    # maxiter=100, popsize=15 might take a while.
    # Let's try a moderate configuration.
    
    best_result = None
    best_val = 1e10
    
    try:
        # Use differential evolution
        # tol and maxiter control convergence
        # strategy='best1bin' is standard
        res = differential_evolution(
            penalty_objective, 
            bounds, 
            maxiter=200, 
            popsize=20, 
            tol=1e-6,
            seed=42,
            polish=True, # Use L-BFGS-B to polish the result
            init='latinhypercube'
        )
        
        if res.success or res.fun < 1e9:
            best_result = res.x
            best_val = res.fun
            
    except Exception as e:
        pass

    # If DE fails or is too slow, try a simpler local search from a good grid start
    if best_result is None:
        # Fallback to a simple grid in a 1x1 square
        # 5x5 grid = 25 cells, take 21
        side = 5
        x0, y0, r0 = [], [], []
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N: break
                x0.append((i + 0.5) / side)
                y0.append((j + 0.5) / side)
                r0.append(0.5 / side * 0.95)
                count += 1
            if count >= N: break
        
        # Optimize this grid with SLSQP (fast local search)
        w0 = 1.0
        x0 = np.array(x0)
        y0 = np.array(y0)
        r0 = np.array(r0)
        vars0 = np.concatenate(([w0], x0, y0, r0))
        
        def objective_slsqp(v):
            return penalty_objective(v)
            
        try:
            res2 = minimize(objective_slsqp, vars0, method='L-BFGS-B', 
                            bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-10})
            if res2.fun < best_val:
                best_result = res2.x
                best_val = res2.fun
        except Exception as e:
            pass

    # Extract result
    if best_result is not None:
        w = best_result[0]
        x = best_result[1:1+N]
        y = best_result[1+N:1+2*N]
        r = best_result[1+2*N:1+3*N]
        
        # Final validation and cleanup
        # Ensure r > 0
        r = np.maximum(r, 1e-6)
        
        # Check validity
        valid = True
        h = 2.0 - w
        for i in range(N):
            if x[i] < r[i] - 1e-9 or x[i] > w - r[i] + 1e-9 or y[i] < r[i] - 1e-9 or y[i] > h - r[i] + 1e-9:
                valid = False
                break
        for i in range(N):
            for j in range(i+1, N):
                dist2 = (x[i]-x[j])**2 + (y[i]-y[j])**2
                sum_r = r[i] + r[j]
                if dist2 < sum_r**2 - 1e-9:
                    valid = False
                    break
            if not valid: break
            
        if valid:
            return np.column_stack((x, y, r))
        else:
            # If invalid, return a known valid fallback (grid in 1x1)
            side = 5
            cells = []
            count = 0
            for i in range(side):
                for j in range(side):
                    if count >= N: break
                    cells.append([(i + 0.5) / side, (j + 0.5) / side, 0.5 / side * 0.9])
                    count += 1
                if count >= N: break
            return np.array(cells)
    else:
        # Fallback grid
        side = 5
        cells = []
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N: break
                cells.append([(i + 0.5) / side, (j + 0.5) / side, 0.5 / side * 0.9])
                count += 1
            if count >= N: break
        return np.array(cells)
