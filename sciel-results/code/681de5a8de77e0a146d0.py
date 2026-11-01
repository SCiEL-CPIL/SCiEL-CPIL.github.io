"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    best_val = 0.0
    best_config = None
    
    # We need to pack 21 circles in a rectangle with perimeter 4.
    # Let width = w, height = h. 2*(w+h) <= 4 => w+h <= 2.
    # We can fix w+h=2 for maximum area (for a given aspect ratio, area is max when w=h? No, area = w*(2-w), max at w=1).
    # But the aspect ratio matters for packing.
    # Let's parameterize by aspect ratio r = w/h.
    # w + h = 2 => h(1+r) = 2 => h = 2/(1+r), w = 2r/(1+r).
    
    # Objective: maximize sum of radii.
    # Constraints:
    # 1. Circles inside rectangle: x_i - r_i >= 0, x_i + r_i <= w, y_i - r_i >= 0, y_i + r_i <= h
    # 2. Circles disjoint: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    
    # This is a non-linear programming problem.
    # Variables: x_1..x_N, y_1..y_N, r_1..r_N, r_aspect (or just w,h with constraint w+h=2)
    # Let's use variables: x, y, r, and log_w (or just w, with h=2-w)
    # To ensure positivity, we can use log variables for radii if needed, but radii are positive.
    
    # Let's try a fixed aspect ratio first, or optimize it too.
    # Optimizing aspect ratio adds a variable. Let's start with w as a variable, h=2-w.
    # w in (0, 2).
    
    # Total variables: 2*N (positions) + N (radii) + 1 (width) = 3N + 1 = 64.
    
    # Objective: -sum(r)
    
    # Constraints:
    # For each circle i:
    #   x_i - r_i >= 0  =>  r_i - x_i <= 0
    #   x_i + r_i <= w  =>  x_i + r_i - w <= 0
    #   y_i - r_i >= 0  =>  r_i - y_i <= 0
    #   y_i + r_i <= h  =>  y_i + r_i - (2-w) <= 0  =>  y_i + r_i + w - 2 <= 0
    
    # For each pair (i,j) with i<j:
    #   (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    #   =>  (r_i + r_j)^2 - (x_i - x_j)^2 - (y_i - y_j)^2 <= 0
    
    # This is a non-convex problem. SLSQP or L-BFGS-B with penalties?
    # SLSQP can handle inequalities.
    
    # Let's use a penalty method with L-BFGS-B or SLSQP.
    # SLSQP is more robust for non-linear constraints.
    
    def objective(vars):
        w = vars[0]
        h = 2.0 - w
        xy = vars[1:1+2*N].reshape(N, 2)
        r = vars[1+2*N:1+3*N]
        
        # Bounds check for w
        if w <= 0 or h <= 0:
            return 1e10
        
        # Objective: -sum(r)
        return -np.sum(r)
    
    def constraints(vars):
        w = vars[0]
        h = 2.0 - w
        xy = vars[1:1+2*N].reshape(N, 2)
        r = vars[1+2*N:1+3*N]
        
        cons = []
        
        # w bounds: 0.1 <= w <= 1.9 (symmetric, so just 0.1 to 1.0? No, w and h interchangeable. Let's fix w >= h to break symmetry? Or just let it be.)
        # Actually, w and h are symmetric. We can constrain 0.5 <= w <= 1.5 to avoid redundancy? No, just let it be.
        
        # Circle-rectangle constraints
        for i in range(N):
            x, y = xy[i]
            ri = r[i]
            # x - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[1+2*i] - v[1+2*N+i]})
            # x + r <= w
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[0] - (v[1+2*i] + v[1+2*N+i])})
            # y - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2+2*i] - v[1+2*N+i]})
            # y + r <= h = 2 - w
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: (2.0 - v[0]) - (v[2+2*i] + v[1+2*N+i])})
        
        # Circle-circle constraints
        for i in range(N):
            for j in range(i+1, N):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda v, i=i, j=j: (v[1+2*i] - v[1+2*j])**2 + (v[2+2*i] - v[2+2*j])**2 - (v[1+2*N+i] + v[1+2*N+j])**2
                })
        
        return cons
    
    # This creates too many constraints for SLSQP to be efficient.
    # Better to use a penalty function.
    
    def penalty_objective(vars, beta):
        w = vars[0]
        h = 2.0 - w
        xy = vars[1:1+2*N].reshape(N, 2)
        r = vars[1+2*N:1+3*N]
        
        if w <= 0 or h <= 0:
            return 1e10
        
        obj = -np.sum(r)
        
        # Penalty for rectangle violations
        pen = 0.0
        for i in range(N):
            x, y = xy[i]
            ri = r[i]
            pen += max(0, ri - x)**2
            pen += max(0, x + ri - w)**2
            pen += max(0, ri - y)**2
            pen += max(0, y + ri - h)**2
        
        # Penalty for circle-circle violations
        for i in range(N):
            for j in range(i+1, N):
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                dist2 = dx*dx + dy*dy
                min_dist = r[i] + r[j]
                if dist2 < min_dist**2:
                    pen += (min_dist**2 - dist2)**2
        
        return obj + beta * pen
    
    # Multi-start
    n_starts = 50
    rng = np.random.default_rng(42)
    
    for start_idx in range(n_starts):
        if time.time() > deadline:
            break
        
        # Random initial aspect ratio
        w0 = rng.uniform(0.5, 1.5)
        h0 = 2.0 - w0
        
        # Initial positions: random within rectangle
        # Initial radii: small, e.g., min(w0, h0) / (2 * sqrt(N))
        r_init = min(w0, h0) / (2.0 * np.sqrt(N)) * 0.9
        r0 = np.full(N, r_init)
        
        # Random positions
        xy0 = np.column_stack([
            rng.uniform(0, w0, N),
            rng.uniform(0, h0, N)
        ])
        
        # Ensure initial feasibility (approximate)
        # Just place them in a grid-like pattern for better start
        cols = int(np.ceil(np.sqrt(N * w0 / h0)))
        rows = int(np.ceil(N / cols))
        
        # Adjust to fit
        # Let's use a grid packing as initial guess
        w_cell = w0 / cols
        h_cell = h0 / rows
        
        # Use the smaller cell dimension for radius
        r_init = min(w_cell, h_cell) / 2.0 * 0.95
        
        # Generate grid positions
        xy_grid = []
        r_grid = []
        for i in range(N):
            col = i % cols
            row = i // cols
            x = (col + 0.5) * w_cell
            y = (row + 0.5) * h_cell
            xy_grid.append([x, y])
            r_grid.append(r_init)
        
        xy0 = np.array(xy_grid)
        r0 = np.array(r_grid)
        
        # Add some random perturbation
        xy0 += rng.normal(0, 0.01 * min(w0, h0), xy0.shape)
        r0 += rng.normal(0, 0.01 * r_init, N)
        r0 = np.maximum(r0, 1e-6)
        
        # Clamp positions to rectangle
        xy0[:, 0] = np.clip(xy0[:, 0], 0, w0)
        xy0[:, 1] = np.clip(xy0[:, 1], 0, h0)
        
        # Build initial vector
        vars0 = np.concatenate([[w0], xy0.flatten(), r0])
        
        # Optimize
        beta = 1.0
        current_vars = vars0.copy()
        
        for beta_iter in range(20):
            if time.time() > deadline:
                break
            
            res = minimize(
                penalty_objective,
                current_vars,
                args=(beta,),
                method='L-BFGS-B',
                bounds=[(0.1, 1.9)] + [(0, 2.0)] * (2*N) + [(1e-6, 1.0)] * N,
                options={'maxiter': 1000, 'ftol': 1e-10}
            )
            
            if res.success:
                current_vars = res.x
                beta *= 10
            else:
                # If failed, try smaller step or break
                if beta > 1e6:
                    break
        
        # Evaluate final configuration
        w = current_vars[0]
        h = 2.0 - w
        xy = current_vars[1:1+2*N].reshape(N, 2)
        r = current_vars[1+2*N:1+3*N]
        
        # Check validity
        valid = True
        for i in range(N):
            x, y = xy[i]
            ri = r[i]
            if x - ri < -1e-6 or x + ri > w + 1e-6 or y - ri < -1e-6 or y + ri > h + 1e-6:
                valid = False
                break
        
        if valid:
            for i in range(N):
                for j in range(i+1, N):
                    dx = xy[i, 0] - xy[j, 0]
                    dy = xy[i, 1] - xy[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < r[i] + r[j] - 1e-6:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            val = np.sum(r)
            if val > best_val:
                best_val = val
                best_config = np.column_stack([xy, r])
        
        # Refine with SLSQP if time permits
        if time.time() < deadline - 10 and valid:
            try:
                res2 = minimize(
                    objective,
                    current_vars,
                    method='SLSQP',
                    bounds=[(0.1, 1.9)] + [(0, 2.0)] * (2*N) + [(1e-6, 1.0)] * N,
                    constraints=constraints,
                    options={'maxiter': 500, 'ftol': 1e-10}
                )
                if res2.success:
                    w = res2.x[0]
                    h = 2.0 - w
                    xy = res2.x[1:1+2*N].reshape(N, 2)
                    r = res2.x[1+2*N:1+3*N]
                    
                    valid2 = True
                    for i in range(N):
                        x, y = xy[i]
                        ri = r[i]
                        if x - ri < -1e-6 or x + ri > w + 1e-6 or y - ri < -1e-6 or y + ri > h + 1e-6:
                            valid2 = False
                            break
                    
                    if valid2:
                        for i in range(N):
                            for j in range(i+1, N):
                                dx = xy[i, 0] - xy[j, 0]
                                dy = xy[i, 1] - xy[j, 1]
                                dist = np.sqrt(dx*dx + dy*dy)
                                if dist < r[i] + r[j] - 1e-6:
                                    valid2 = False
                                    break
                            if not valid2:
                                break
                    
                    if valid2:
                        val2 = np.sum(r)
                        if val2 > best_val:
                            best_val = val2
                            best_config = np.column_stack([xy, r])
            except Exception:
                pass
    
    if best_config is None:
        # Fallback: grid packing
        side = int(np.ceil(np.sqrt(N)))
        # Use square aspect ratio w=h=1
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_config = np.array(cells[:N])
    
    return best_config
# EVOLVE-BLOCK-END
