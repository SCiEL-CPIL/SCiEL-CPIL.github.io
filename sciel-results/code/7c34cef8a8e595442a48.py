"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    """
    Pack 21 circles in a rectangle of perimeter <= 4, maximizing sum of radii.
    Uses SLSQP to jointly optimize circle centers, radii, and rectangle aspect ratio.
    """
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s limit
    
    def elapsed():
        return time.time() - start_time
    
    # Decision variables: for each circle (cx, cy, r), plus rectangle width w (height = 2 - w)
    # Total variables: 21*3 + 1 = 64
    
    # Initial guess: grid-like placement
    # Try a few different initial configurations and pick the best
    
    best_x = None
    best_val = -np.inf
    
    # Helper to build the optimization problem from an initial guess
    def run_optimization(x0, max_iter=500):
        n = N
        # x0 layout: [cx_0, cy_0, r_0, ..., cx_n-1, cy_n-1, r_n-1, w]
        # Rectangle: width = w, height = 2 - w (perimeter = 2*(w + (2-w)) = 4)
        
        def objective(x):
            # Negative sum of radii (we maximize sum of radii)
            radii = x[3*n:3*n+n]
            return -np.sum(radii)
        
        def constraints(x):
            cons = []
            cx = x[0:n]
            cy = x[n:2*n]
            r = x[3*n:3*n+n]
            w = x[3*n+n]
            h = 2.0 - w
            
            # Bounds: circles inside rectangle
            # cx - r >= 0, cx + r <= w
            # cy - r >= 0, cy + r <= h
            cons.append({'type': 'ineq', 'fun': lambda x: x[0:n] - x[3*n:3*n+n]})  # cx - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda x: x[3*n+n] - x[0:n] - x[3*n:3*n+n]})  # w - cx - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda x: x[n:2*n] - x[3*n:3*n+n]})  # cy - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda x: (2.0 - x[3*n+n]) - x[n:2*n] - x[3*n:3*n+n]})  # h - cy - r >= 0
            
            # Minimum radius
            cons.append({'type': 'ineq', 'fun': lambda x: x[3*n:3*n+n] - 0.01})  # r >= 0.01
            
            # Aspect ratio bounds: w in [0.5, 1.5] (so h in [0.5, 1.5])
            cons.append({'type': 'ineq', 'fun': lambda x: x[3*n+n] - 0.3})
            cons.append({'type': 'ineq', 'fun': lambda x: 1.7 - x[3*n+n]})
            
            # Circle-circle non-overlap: (cx_i - cx_j)^2 + (cy_i - cy_j)^2 >= (r_i + r_j)^2
            # This is O(n^2) constraints
            for i in range(n):
                for j in range(i+1, n):
                    cons.append({
                        'type': 'ineq',
                        'fun': lambda x, i=i, j=j: 
                            (x[i] - x[j])**2 + (x[n+i] - x[n+j])**2 - 
                            (x[3*n+i] + x[3*n+j])**2
                    })
            
            return cons
        
        bounds = []
        for i in range(n):
            bounds.append((0.01, 2.0))  # cx
        for i in range(n):
            bounds.append((0.01, 2.0))  # cy
        for i in range(n):
            bounds.append((0.01, 1.0))  # r
        bounds.append((0.3, 1.7))  # w
        
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                      constraints=constraints, options={'maxiter': max_iter, 'ftol': 1e-8})
        return res
    
    # Generate several initial configurations
    initial_configs = []
    
    # Config 1: Grid 5x5 (25 cells, use 21)
    side = 5
    w0 = 1.0
    h0 = 1.0
    cells = []
    for i in range(side):
        for j in range(side):
            cx = (i + 0.5) * w0 / side
            cy = (j + 0.5) * h0 / side
            r = min(w0, h0) / side * 0.49
            cells.append((cx, cy, r))
    initial_configs.append(cells[:N])
    
    # Config 2: Grid 4x6 (24 cells, use 21) with wider rectangle
    w0 = 1.3
    h0 = 0.7
    side_x = 6
    side_y = 4
    cells = []
    for i in range(side_x):
        for j in range(side_y):
            cx = (i + 0.5) * w0 / side_x
            cy = (j + 0.5) * h0 / side_y
            r = min(w0/side_x, h0/side_y) * 0.49
            cells.append((cx, cy, r))
    initial_configs.append(cells[:N])
    
    # Config 3: Grid 6x4 with taller rectangle
    w0 = 0.7
    h0 = 1.3
    side_x = 4
    side_y = 6
    cells = []
    for i in range(side_x):
        for j in range(side_y):
            cx = (i + 0.5) * w0 / side_x
            cy = (j + 0.5) * h0 / side_y
            r = min(w0/side_x, h0/side_y) * 0.49
            cells.append((cx, cy, r))
    initial_configs.append(cells[:N])
    
    # Config 4: Square grid 5x5 with slightly different spacing
    side = 5
    w0 = 1.0
    h0 = 1.0
    cells = []
    for i in range(side):
        for j in range(side):
            cx = (i + 0.5) * w0 / side
            cy = (j + 0.5) * h0 / side
            r = min(w0, h0) / side * 0.45
            cells.append((cx, cy, r))
    initial_configs.append(cells[:N])
    
    # Config 5: Hexagonal-ish packing
    w0 = 1.0
    h0 = 1.0
    cells = []
    rows = 5
    cols = 5
    for j in range(rows):
        for i in range(cols):
            cx = (i + 0.5) * w0 / cols
            if j % 2 == 1:
                cx += w0 / cols * 0.3
            cy = (j + 0.5) * h0 / rows
            r = min(w0/cols, h0/rows) * 0.42
            if 0 < cx < w0 and 0 < cy < h0:
                cells.append((cx, cy, r))
    initial_configs.append(cells[:N])
    
    # Run optimization from each initial config
    for idx, config in enumerate(initial_configs):
        if elapsed() > deadline - 5:
            break
        
        # Build x0 vector
        x0 = []
        for (cx, cy, r) in config:
            x0.extend([cx, cy, r])
        x0.append(1.0)  # initial width
        x0 = np.array(x0)
        
        try:
            res = run_optimization(x0, max_iter=300)
            if res.success or res.fun < best_val:
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_x = res.x
        except Exception as e:
            pass
        
        if elapsed() > deadline - 10:
            break
    
    # If no good result, fall back to simple grid
    if best_x is None:
        # Simple grid fallback
        side = 5
        w = 1.0
        h = 1.0
        result = []
        for i in range(side):
            for j in range(side):
                if len(result) >= N:
                    break
                cx = (i + 0.5) * w / side
                cy = (j + 0.5) * h / side
                r = min(w, h) / side * 0.49
                result.append([cx, cy, r])
        return np.array(result)
    
    # Extract results
    cx = best_x[0:N]
    cy = best_x[N:2*N]
    r = best_x[3*N:3*N+N]
    w = best_x[3*N+N]
    h = 2.0 - w
    
    # Clamp to ensure validity
    result = []
    for i in range(N):
        cx_i = np.clip(cx[i], r[i], w - r[i])
        cy_i = np.clip(cy[i], r[i], h - r[i])
        r_i = max(r[i], 0.01)
        result.append([cx_i, cy_i, r_i])
    
    # Final verification and adjustment if needed
    result_arr = np.array(result)
    
    # Check for overlaps and fix if necessary
    for iteration in range(100):
        if elapsed() > deadline:
            break
        overlap_fixed = False
        for i in range(N):
            for j in range(i+1, N):
                dx = result_arr[i, 0] - result_arr[j, 0]
                dy = result_arr[i, 1] - result_arr[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = result_arr[i, 2] + result_arr[j, 2]
                if dist < min_dist - 1e-6:
                    # Push apart
                    if dist < 1e-8:
                        dx, dy = 1.0, 0.0
                        dist = 1.0
                    push = (min_dist - dist) / 2.0 + 1e-4
                    dx_norm = dx / dist
                    dy_norm = dy / dist
                    result_arr[i, 0] += push * dx_norm
                    result_arr[i, 1] += push * dy_norm
                    result_arr[j, 0] -= push * dx_norm
                    result_arr[j, 1] -= push * dy_norm
                    overlap_fixed = True
        if not overlap_fixed:
            break
    
    # Clamp again after overlap fixing
    for i in range(N):
        r_i = result_arr[i, 2]
        result_arr[i, 0] = np.clip(result_arr[i, 0], r_i, w - r_i)
        result_arr[i, 1] = np.clip(result_arr[i, 1], r_i, h - r_i)
    
    return result_arr

# EVOLVE-BLOCK-END
