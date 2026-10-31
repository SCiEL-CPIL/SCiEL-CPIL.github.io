"""Circle packing in perimeter-4 rectangle: SLSQP Global Optimization with Multi-Start."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _is_valid(cx, cy, r, W, H, tol=1e-7):
    """Check if packing is valid."""
    if W <= 0.01 or H <= 0.01:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True

def _objective_and_constraints(vars):
    """
    vars: [x_0..x_N-1, y_0..y_N-1, r_0..r_N-1, W, H]
    Objective: Minimize -sum(r)
    Constraints:
      1. W + H = 2 (Equality)
      2. x_i - r_i >= 0 (Inequality, >= 0 form for scipy)
      3. W - x_i - r_i >= 0
      4. y_i - r_i >= 0
      5. H - y_i - r_i >= 0
      6. dist(i,j) - (r_i + r_j) >= 0
    """
    x = vars[0:N]
    y = vars[N:2*N]
    r = vars[2*N:3*N]
    W = vars[3*N]
    H = vars[3*N+1]
    
    # Objective: maximize sum(r) -> minimize -sum(r)
    obj = -np.sum(r)
    
    # Constraints list: fun(x) >= 0
    cons = []
    
    # 1. Perimeter constraint: W + H = 2
    cons.append({
        'type': 'eq',
        'fun': lambda v: v[3*N] + v[3*N+1] - 2.0
    })
    
    # 2-5. Wall constraints
    # Left wall: x_i - r_i >= 0
    cons.append({
        'type': 'ineq',
        'fun': lambda v: v[0:N] - v[2*N:3*N]
    })
    # Right wall: W - x_i - r_i >= 0
    cons.append({
        'type': 'ineq',
        'fun': lambda v: v[3*N] - v[0:N] - v[2*N:3*N]
    })
    # Bottom wall: y_i - r_i >= 0
    cons.append({
        'type': 'ineq',
        'fun': lambda v: v[N:2*N] - v[2*N:3*N]
    })
    # Top wall: H - y_i - r_i >= 0
    cons.append({
        'type': 'ineq',
        'fun': lambda v: v[3*N+1] - v[N:2*N] - v[2*N:3*N]
    })
    
    # 6. Pairwise separation: dist(i,j) >= r_i + r_j
    # We create individual constraints for each pair to help the solver
    # This is O(N^2) constraints, which is 210 for N=21. Manageable for SLSQP.
    for i in range(N):
        for j in range(i + 1, N):
            # dist(i,j) - r_i - r_j >= 0
            # dist = sqrt((x_i-x_j)^2 + (y_i-y_j)^2)
            # We define a function for this specific pair
            # Note: Using lambdas inside loops can be slow, but N=21 is small.
            # To be safer/faster, we could vectorize, but SLSQP expects scalar constraints usually.
            # Let's try vectorizing the pairwise constraints into a single array constraint if possible?
            # scipy minimize supports array constraints.
            pass # We'll handle this differently below for efficiency.

    # Actually, creating 210 individual constraint dicts is heavy.
    # Let's use a single vector constraint for all pairs.
    # Constraint: d_ij - r_i - r_j >= 0 for all i<j
    # We can return an array of length N*(N-1)/2.
    
    def pairwise_constraints(v):
        x_v = v[0:N]
        y_v = v[N:2*N]
        r_v = v[2*N:3*N]
        cons_vals = []
        for i in range(N):
            for j in range(i + 1, N):
                dx = x_v[i] - x_v[j]
                dy = y_v[i] - y_v[j]
                d = np.sqrt(dx*dx + dy*dy)
                cons_vals.append(d - r_v[i] - r_v[j])
        return np.array(cons_vals)
        
    cons.append({
        'type': 'ineq',
        'fun': pairwise_constraints
    })
    
    return obj, cons

def _polish_radii(x, y, W, H, r_init):
    """
    Given fixed positions and box, maximize radii.
    This is a linear programming problem if we ignore the sqrt in distance? No, distance is fixed.
    Constraints:
      r_i <= x_i
      r_i <= W - x_i
      r_i <= y_i
      r_i <= H - y_i
      r_i + r_j <= dist(i,j)
    Maximize sum(r)
    
    We can use scipy.optimize.linprog or simply iterative projection.
    Iterative projection is robust and fast for small N.
    """
    r = r_init.copy()
    # Precompute distances
    d_mat = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d_mat[i,j] = np.sqrt(dx*dx + dy*dy)
                
    # Wall limits
    wall_lim = np.minimum(np.minimum(x, W-x), np.minimum(y, H-y))
    
    for _ in range(100):
        improved = False
        for i in range(N):
            # Max radius allowed by walls
            max_r = wall_lim[i]
            
            # Max radius allowed by neighbors
            for j in range(N):
                if i != j:
                    # r_i <= d_ij - r_j
                    allowed = d_mat[i,j] - r[j]
                    if allowed < max_r:
                        max_r = allowed
                        
            if max_r < 0:
                max_r = 1e-9 # Should not happen if valid, but safety
                
            if max_r > r[i] + 1e-10:
                r[i] = max_r
                improved = True
        if not improved:
            break
            
    return r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_value = 0.0
    best_result = None
    
    def update_best(x, y, r, W, H):
        nonlocal best_value, best_result
        # Validate
        if _is_valid(x, y, r, W, H):
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                best_result = (x.copy(), y.copy(), r.copy(), W, H)
                
    # Define initializations
    # We will try several aspect ratios and random perturbations
    # W + H = 2. Let W = 1 + delta, H = 1 - delta. delta in [-0.7, 0.7]
    
    # Strategy:
    # 1. Generate a few diverse starting points.
    # 2. Run SLSQP from each.
    # 3. Polish the best ones.
    
    # To save time, we limit the number of SLSQP runs.
    # SLSQP is slow. Maybe 5-10 runs?
    
    # Initial guesses for W, H
    aspect_deltas = [0.0, 0.2, -0.2, 0.4, -0.4, 0.6, -0.6, 0.1, -0.1]
    
    # We will use a "grid + noise" initialization for x,y,r
    # Grid size roughly sqrt(21) ~ 4.6 -> 5x5 or 4x6
    
    attempts = 0
    max_attempts = 15 # Limit SLSQP calls
    
    for delta in aspect_deltas:
        if time.time() > deadline - 10:
            break
            
        W = 1.0 + delta
        H = 1.0 - delta
        
        # Skip if dimensions are too small
        if W < 0.3 or H < 0.3:
            continue
            
        # Create initial grid
        # For 21 circles, a 5x5 grid has 25 spots, remove 4 corners?
        # Or 4x6 grid?
        # Let's use a simple hex-like packing initialization
        
        # Estimate radius
        # Area = W*H. 21 circles. r ~ sqrt(Area / (21 * 2*sqrt(3)))
        r_est = np.sqrt((W * H) / (N * 2 * np.sqrt(3)))
        
        # Grid placement
        cols = 5
        rows = 5
        # Positions in a 5x5 grid
        xs = []
        ys = []
        for i in range(5):
            for j in range(5):
                # Center of cell
                cx = (i + 0.5) * (W / 5.0)
                cy = (j + 0.5) * (H / 5.0)
                xs.append(cx)
                ys.append(cy)
        
        # We have 25 positions, need 21. Remove 4 corners to maximize interior space?
        # Or just pick the first 21?
        # Better: Pick positions that are more central.
        # Let's just use the first 21 for simplicity, but add noise.
        
        x0 = np.array(xs[:21])
        y0 = np.array(ys[:21])
        
        # Add small random noise to break symmetry
        rng = np.random.default_rng(attempts)
        x0 = x0 + rng.uniform(-0.01, 0.01, N)
        y0 = y0 + rng.uniform(-0.01, 0.01, N)
        
        # Clip to box with margin
        x0 = np.clip(x0, r_est*0.5, W - r_est*0.5)
        y0 = np.clip(y0, r_est*0.5, H - r_est*0.5)
        
        r0 = np.full(N, r_est * 0.8) # Start slightly smaller to allow growth
        
        # Initial variables vector
        x0_vars = np.concatenate([x0, y0, r0, [W, H]])
        
        # Bounds for SLSQP
        # x, y, r must be positive. W, H positive.
        # r > 0, x > 0, y > 0, W > 0.3, H > 0.3
        bounds = []
        for i in range(N):
            bounds.append((1e-6, None)) # x_i
        for i in range(N):
            bounds.append((1e-6, None)) # y_i
        for i in range(N):
            bounds.append((1e-6, None)) # r_i
        bounds.append((0.3, 1.7)) # W
        bounds.append((0.3, 1.7)) # H
        
        # Run SLSQP
        try:
            res = minimize(
                lambda v: _objective_and_constraints(v)[0],
                x0_vars,
                method='SLSQP',
                bounds=bounds,
                constraints=_objective_and_constraints(x0_vars)[1],
                options={'maxiter': 200, 'ftol': 1e-9}
            )
            
            if res.success:
                x_res = res.x[0:N]
                y_res = res.x[N:2*N]
                r_res = res.x[2*N:3*N]
                W_res = res.x[3*N]
                H_res = res.x[3*N+1]
                
                # Polish radii
                r_polished = _polish_radii(x_res, y_res, W_res, H_res, r_res)
                
                update_best(x_res, y_res, r_polished, W_res, H_res)
            else:
                # Even if not success, check if result is better
                x_res = res.x[0:N]
                y_res = res.x[N:2*N]
                r_res = res.x[2*N:3*N]
                W_res = res.x[3*N]
                H_res = res.x[3*N+1]
                
                # Try to polish anyway
                r_polished = _polish_radii(x_res, y_res, W_res, H_res, r_res)
                update_best(x_res, y_res, r_polished, W_res, H_res)
                
        except Exception as e:
            pass # Ignore failures
            
        attempts += 1
        
        # If we have time, try a few random restarts from the best solution found so far
        if best_result is not None and attempts < max_attempts and time.time() < deadline - 20:
            x_best, y_best, r_best, W_best, H_best = best_result
            for restart in range(2):
                if time.time() > deadline - 5:
                    break
                # Perturb best solution
                x_pert = x_best + np.random.uniform(-0.05, 0.05, N)
                y_pert = y_best + np.random.uniform(-0.05, 0.05, N)
                r_pert = r_best * 0.9 # Shrink to allow movement
                W_pert = W_best
                H_pert = H_best
                
                x_pert = np.clip(x_pert, r_pert, W_pert - r_pert)
                y_pert = np.clip(y_pert, r_pert, H_pert - r_pert)
                
                x0_vars = np.concatenate([x_pert, y_pert, r_pert, [W_pert, H_pert]])
                
                try:
                    res = minimize(
                        lambda v: _objective_and_constraints(v)[0],
                        x0_vars,
                        method='SLSQP',
                        bounds=bounds,
                        constraints=_objective_and_constraints(x0_vars)[1],
                        options={'maxiter': 100, 'ftol': 1e-9}
                    )
                    if res.success or res.fun < 0: # If it reduced objective (increased sum r)
                        x_res = res.x[0:N]
                        y_res = res.x[N:2*N]
                        r_res = res.x[2*N:3*N]
                        W_res = res.x[3*N]
                        H_res = res.x[3*N+1]
                        
                        r_polished = _polish_radii(x_res, y_res, W_res, H_res, r_res)
                        update_best(x_res, y_res, r_polished, W_res, H_res)
                except:
                    pass
                attempts += 1

    # Fallback if no valid result found
    if best_result is None:
        W, H = 1.0, 1.0
        # Simple grid
        x = np.linspace(0.1, 0.9, N)
        y = np.linspace(0.1, 0.9, N)
        # This is not a valid packing, but we need to return something.
        # Let's use the hexagonal init from the seed as a fallback.
        # Re-implementing hex init quickly for fallback
        cols = 5
        rows = 5
        xs = []
        ys = []
        for i in range(5):
            for j in range(5):
                cx = (i + 0.5) * (W / 5.0)
                cy = (j + 0.5) * (H / 5.0)
                xs.append(cx)
                ys.append(cy)
        x = np.array(xs[:21])
        y = np.array(ys[:21])
        r = np.full(N, 0.1)
        best_result = (x, y, r, W, H)
        
    x, y, r, W, H = best_result
    
    # Final safety shrink
    for _ in range(3):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, W, H):
            r = r_trial
        else:
            break
            
    return np.column_stack((x, y, r))
