import numpy as np
from scipy.optimize import minimize

N = 26

def construct():
    # 1. Initialize with a hexagonal pattern
    # Approximate hexagonal packing for 26 circles
    # We will use a grid and then optimize
    
    # Initial guess: Grid
    # 5x5 grid = 25, add 1 more? 
    # Let's try 6 rows.
    # Row lengths: 5, 4, 5, 4, 5, 3 ? Total 26.
    
    # Simpler: Start with a 5x5 grid (25 circles) and 1 extra, or just 26 in a distorted grid.
    # Let's use a 5x5 grid for the first 25 and place the 26th in the center of a 2x2 block?
    # Or just use a 5x5 grid and make one row have 6? No, 5x5=25.
    # Let's use 4 rows of 7? 28. Too many.
    # 5 rows of 5 = 25. 
    # Let's use 6 rows.
    # 5, 5, 5, 5, 5, 1?
    
    # Let's just create a random valid starting point or a grid and optimize.
    # Grid initialization:
    n_rows = 5
    n_cols = 6 # 30 circles, take 26?
    # Better: 5x5 grid (25) + 1 circle in the middle of one of the cells?
    
    # Let's stick to a 5x5 grid for the first 25, and put the 26th in the center of the square?
    # That might cause overlaps.
    
    # Let's use a 6x5 grid (30 circles) and remove 4?
    # Or simply generate 26 points on a grid 5x6 and optimize.
    
    # Let's use a 5x5 grid (25 circles) and add one more in a "gap".
    # Actually, the simplest robust start is a grid of 26 circles.
    # 5 rows, 6 cols? 30 circles. 
    # Let's use 4 rows, 7 cols? 28.
    # Let's use 5 rows, 5 cols (25) and 1 extra.
    
    # Let's try a hexagonal packing initialization.
    # Hexagonal packing of 26 circles.
    # Rows: 5, 5, 5, 5, 5, 1?
    
    # Let's just use a grid of 26 circles: 5 rows, 6 columns, but only 26 used?
    # No, let's use a 5x5 grid (25) and place the 26th at (0.5, 0.5) with small radius?
    # That's bad.
    
    # Let's use a 6x5 grid (30 circles) and randomly pick 26? No.
    
    # Let's use a 5x5 grid (25 circles) and add one in the center of the top row?
    
    # Okay, let's use a standard grid of 26 circles:
    # 5 rows. 
    # Cols: 6, 5, 6, 5, 4? Total 26.
    
    # Let's just use a 5x5 grid (25) and one more circle placed at (0.5, 0.5) 
    # and let the optimizer figure it out.
    
    # Actually, a better start is a hexagonal lattice.
    # Let's generate a hexagonal lattice of ~26 points.
    
    def hex_packing(n):
        # Approximate number of circles in hexagonal packing of n
        # We want to fit them in [0,1]^2.
        # Let's estimate the radius.
        # Area = n * pi * r^2 <= 1
        # r <= 1/sqrt(n*pi)
        
        # Let's try to fit a 5x5 hex grid.
        # Row 0: 5 circles
        # Row 1: 5 circles (shifted)
        # Row 2: 5 circles
        # Row 3: 5 circles
        # Row 4: 5 circles
        # Total 25.
        # Add 1 more?
        
        # Let's use 6 rows.
        # 5, 4, 5, 4, 5, 3? Total 26.
        
        rows = [5, 4, 5, 4, 5, 3]
        circles = []
        y_spacing = 1.0 / (len(rows) * 0.866) # Hex vertical spacing
        # This is getting complicated.
        
        # Let's just use a grid and optimize.
        # 5x5 grid (25) + 1 extra.
        # Or 6x5 grid (30) -> take 26.
        
        # Let's use 5 rows, 6 cols (30 circles) and take the first 26.
        # This creates a rectangular block of 26 circles.
        # 5 rows, 6 cols.
        # Indices:
        # (0,0) to (0,5)
        # (1,0) to (1,5)
        # (2,0) to (2,5)
        # (3,0) to (3,5)
        # (4,0) to (4,5)
        # That's 30.
        # We take 26.
        # Let's take a 5x5 block (25) and one more?
        
        # Let's just use a 5x5 grid for 25 circles and place the 26th in the center.
        # The optimizer will move them.
        
        return None

    # Let's use a simple grid of 26 circles.
    # 5 rows, 6 cols.
    # We will use the first 26 positions in a 6x5 grid (row-major).
    # 6 columns, 5 rows.
    # 6*5 = 30.
    # We take 26.
    
    x = []
    y = []
    r = []
    
    n_cols = 6
    n_rows = 5
    count = 0
    for i in range(n_rows):
        for j in range(n_cols):
            if count >= N:
                break
            x.append((j + 0.5) / n_cols)
            y.append((i + 0.5) / n_rows)
            r.append(0.5 / n_cols) # Initial radius estimate
            count += 1
        if count >= N:
            break
            
    # Ensure we have exactly N
    while len(x) < N:
        x.append(0.5)
        y.append(0.5)
        r.append(0.05)
        
    x = np.array(x[:N])
    y = np.array(y[:N])
    r = np.array(r[:N])
    
    # Flatten for optimization
    # Variables: x1, y1, r1, x2, y2, r2, ...
    # Total 3*N variables.
    
    def energy(params):
        x_ = params[0::3]
        y_ = params[1::3]
        r_ = params[2::3]
        
        # 1. Boundary violations
        # r <= x, r <= 1-x, r <= y, r <= 1-y
        # If violated, add penalty
        pen_bound = 0.0
        for i in range(N):
            pen_bound += max(0, r_[i] - x_[i])**2
            pen_bound += max(0, r_[i] - (1 - x_[i]))**2
            pen_bound += max(0, r_[i] - y_[i])**2
            pen_bound += max(0, r_[i] - (1 - y_[i]))**2
            
        # 2. Overlap violations
        # (ri + rj)^2 > (xi-xj)^2 + (yi-yj)^2
        pen_overlap = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dist_sq = (x_[i] - x_[j])**2 + (y_[i] - y_[j])**2
                min_dist = r_[i] + r_[j]
                if dist_sq < min_dist**2:
                    # Penalty proportional to overlap
                    overlap = min_dist - np.sqrt(dist_sq)
                    pen_overlap += overlap**2
                    
        # Objective: Maximize sum(r) -> Minimize -sum(r)
        # Total Energy = -sum(r) + lambda * (pen_bound + pen_overlap)
        
        # We need to tune lambda.
        # If lambda is too high, radii will be small to avoid penalties.
        # If lambda is too low, circles will overlap.
        
        # Let's use a large lambda to enforce validity strongly.
        lam = 100.0
        
        return -np.sum(r_) + lam * (pen_bound + pen_overlap)
        
    def grad_energy(params):
        # Numerical gradient
        eps = 1e-7
        grad = np.zeros_like(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += eps
            e_plus = energy(params_plus)
            
            params_minus = params.copy()
            params_minus[i] -= eps
            e_minus = energy(params_minus)
            
            grad[i] = (e_plus - e_minus) / (2 * eps)
        return grad
        
    # Initial parameters
    params0 = np.zeros(3*N)
    params0[0::3] = x
    params0[1::3] = y
    params0[2::3] = r
    
    # Optimization
    # Use L-BFGS-B
    # Bounds:
    # x, y in [0, 1]
    # r in [0, 0.5]
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
        bounds.append((0, 0.5)) # r
        
    # Run optimization
    # Time limit: 120s. 
    # L-BFGS-B might be slow. 
    # Let's try a few iterations.
    
    # To speed up, we can use a simpler local search or reduce iterations.
    # Let's try `scipy.optimize.minimize` with 'L-BFGS-B'.
    
    res = minimize(energy, params0, method='L-BFGS-B', bounds=bounds, 
                   options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-4})
    
    params_opt = res.x
    x_opt = params_opt[0::3]
    y_opt = params_opt[1::3]
    r_opt = params_opt[2::3]
    
    # Post-processing: Ensure strict validity
    # 1. Clamp positions
    # 2. Reduce radii if necessary
    
    # Iterate to resolve any remaining overlaps
    for _ in range(10):
        changed = False
        for i in range(N):
            # Check boundary
            r_max_bound = min(x_opt[i], 1-x_opt[i], y_opt[i], 1-y_opt[i])
            if r_opt[i] > r_max_bound:
                r_opt[i] = r_max_bound
                changed = True
                
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((x_opt[i]-x_opt[j])**2 + (y_opt[i]-y_opt[j])**2)
                min_dist = r_opt[i] + r_opt[j]
                if dist < min_dist:
                    # Scale down radii
                    # Simple fix: reduce both
                    overlap = min_dist - dist
                    # Reduce each by half the overlap?
                    # Or reduce the larger one?
                    # Let's reduce both proportionally to their radii
                    total_r = r_opt[i] + r_opt[j]
                    if total_r > 0:
                        scale = dist / total_r
                        r_opt[i] *= scale
                        r_opt[j] *= scale
                        changed = True
        if not changed:
            break
            
    # Final check and clamp
    for i in range(N):
        r_opt[i] = min(r_opt[i], x_opt[i], 1-x_opt[i], y_opt[i], 1-y_opt[i])
        
    # Construct result
    result = np.zeros((N, 3))
    result[:, 0] = x_opt
    result[:, 1] = y_opt
    result[:, 2] = r_opt
    
    return result
