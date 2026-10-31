"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21
import numpy as np
import time
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    best_sum_r = 0.0
    best_result = None
    
    # We search over aspect ratios.
    # Perimeter 4 => 2(w+h)=4 => h = 2-w.
    # Due to symmetry, we can assume w <= h => w <= 1.
    # But to be safe and cover all, we can just let the optimizer find w.
    # However, fixing w simplifies the search space slightly if we do multi-start on w.
    
    # Let's try a range of w values and for each, run a local optimization.
    # Or, better, include w in the optimization variables.
    
    # Variables: x1, y1, r1, ..., xN, yN, rN, w
    # h = 2 - w
    
    def objective_and_constraints(v):
        # v: [x1, y1, r1, ..., xN, yN, rN, w]
        w = v[-1]
        h = 2.0 - w
        
        # Extract positions and radii
        # Indices 0 to 3N-1 are x,y,r
        # Index 3N is w
        
        if w <= 0 or h <= 0:
            return 1e10, []
            
        xs = v[0:3*N:3]
        ys = v[1:3*N:3]
        rs = v[2:3*N:3]
        
        # Objective: Minimize -sum(rs)
        obj = -np.sum(rs)
        
        constraints = []
        
        # Wall constraints
        for i in range(N):
            # x_i >= r_i
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i] - v[3*i+2]})
            # x_i <= w - r_i  => w - x_i - r_i >= 0
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[-1] - v[3*i] - v[3*i+2]})
            # y_i >= r_i
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]})
            # y_i <= h - r_i  => (2-w) - y_i - r_i >= 0
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 2.0 - v[-1] - v[3*i+1] - v[3*i+2]})
            
        # Circle-circle constraints
        for i in range(N):
            for j in range(i+1, N):
                # (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
                def circ_dist(v, i=i, j=j):
                    xi = v[3*i]
                    yi = v[3*i+1]
                    ri = v[3*i+2]
                    xj = v[3*j]
                    yj = v[3*j+1]
                    rj = v[3*j+2]
                    return (xi-xj)**2 + (yi-yj)**2 - (ri+rj)**2
                constraints.append({'type': 'ineq', 'fun': circ_dist})
                
        # Bounds
        # w in [0.5, 1.5] (symmetry allows [0.5, 1.0] if we swap axes, but let's be broad)
        # r_i >= 0
        # x_i, y_i within some reasonable range, e.g., [0, 2]
        
        bounds = []
        for i in range(N):
            bounds.append((0.0, 2.0)) # x
            bounds.append((0.0, 2.0)) # y
            bounds.append((0.0, 1.0)) # r
        bounds.append((0.5, 1.5)) # w
        
        return obj, constraints, bounds

    # To speed up, we can define the constraint functions more efficiently or use a penalty method.
    # SLSQP with 294 constraints might be slow.
    # Let's try a penalty method with Gradient Descent or Nelder-Mead.
    # Or, simplify by fixing w and optimizing positions/radii.
    
    # Alternative: Fixed w, optimize x,y,r.
    # This reduces variables to 3N=63. Constraints are similar.
    
    def solve_for_w(w_val, init_xs, init_ys, init_rs):
        # v: [x1, y1, r1, ..., xN, yN, rN]
        def obj_pen(v):
            xs = v[0:3*N:3]
            ys = v[1:3*N:3]
            rs = v[2:3*N:3]
            h_val = 2.0 - w_val
            
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
                        # Penalty for overlap
                        # Use a smooth penalty
                        diff = min_dist**2 - dist_sq
                        penalty += diff**2
            
            return -np.sum(rs) + 100.0 * penalty
            
        v0 = np.concatenate([init_xs, init_ys, init_rs])
        
        # Use Nelder-Mead or BFGS (if gradient available)
        # Nelder-Mead is robust for non-smooth penalties
        res = minimize(obj_pen, v0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5})
        
        if res.success or res.fun < 1e-5:
            return res.fun, res.x
        else:
            # Check if the result is valid despite non-success
            # If penalty is low, it's good
            if res.fun < 1e-2:
                return res.fun, res.x
            return None, None

    # Initialization strategies
    # 1. Grid
    # 2. Hexagonal
    
    # Let's sample a few aspect ratios
    w_values = np.linspace(0.6, 1.0, 10) # w <= 1.0 due to symmetry h >= w
    
    for w in w_values:
        if time.time() > deadline:
            break
            
        h = 2.0 - w
        
        # Initialize with a grid
        # Estimate number of circles per row/col
        # Area of rect = w*h. Area of 1 circle ~ pi*r^2.
        # N*pi*r^2 < w*h => r < sqrt(w*h/(N*pi))
        
        # Let's just place them in a grid and set r to a small value, then let optimizer grow.
        # Or estimate r.
        
        # Grid init
        # Try different grid splits
        best_for_w = None
        best_val = -np.inf
        
        # Try splits 4x6, 5x5, etc.
        splits = [(4, 6), (5, 5), (6, 4), (3, 7), (7, 3), (4, 5), (5, 4)]
        
        for rows, cols in splits:
            if time.time() > deadline:
                break
                
            # Create grid points
            xs = []
            ys = []
            # Center of each cell
            for i in range(rows):
                for j in range(cols):
                    x = (j + 0.5) * w / cols
                    y = (i + 0.5) * h / rows
                    xs.append(x)
                    ys.append(y)
            
            # We have rows*cols points. We need N=21.
            # If rows*cols > 21, take first 21.
            # If rows*cols < 21, this split is invalid or we need to add more?
            # For 21, 4x6=24, 5x5=25. These are good.
            
            if len(xs) < N:
                continue
                
            xs = np.array(xs[:N])
            ys = np.array(ys[:N])
            
            # Initial radii
            # Estimate max possible r for this grid
            # Min distance between centers
            min_dist = np.inf
            for i in range(N):
                for j in range(i+1, N):
                    d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                    if d < min_dist:
                        min_dist = d
            # Also distance to walls
            for i in range(N):
                d_w = min(xs[i], w-xs[i], ys[i], h-ys[i])
                if d_w < min_dist:
                    min_dist = d_w
            
            r_init = min_dist / 2.0 * 0.9 # Slightly smaller to start
            rs = np.full(N, r_init)
            
            # Run optimizer
            val, res_x = solve_for_w(w, xs, ys, rs)
            
            if val is not None and val > best_val:
                best_val = val
                best_for_w = res_x
        
        if best_for_w is not None:
            # Extract result
            xs = best_for_w[0:3*N:3]
            ys = best_for_w[1:3*N:3]
            rs = best_for_w[2:3*N:3]
            
            sum_r = np.sum(rs)
            if sum_r > best_sum_r:
                best_sum_r = sum_r
                best_result = np.column_stack([xs, ys, rs])
        
    if best_result is None:
        # Fallback to simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_result

# EVOLVE-BLOCK-END
