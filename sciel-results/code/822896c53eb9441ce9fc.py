# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

TOL = 1e-7
DEADLINE_MARGIN = 15.0

def _is_valid(centers, sides, n=10):
    """Check if a packing is valid."""
    if centers.shape != (n, 2) or sides.shape != (n,):
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    for i in range(n):
        x0 = centers[i, 0] - sides[i] / 2
        x1 = centers[i, 0] + sides[i] / 2
        y0 = centers[i, 1] - sides[i] / 2
        y1 = centers[i, 1] + sides[i] / 2
        if x0 < -TOL - 1e-10 or x1 > 1 + TOL + 1e-10:
            return False
        if y0 < -TOL - 1e-10 or y1 > 1 + TOL + 1e-10:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2 - TOL
            if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                return False
    return True

def _nlp_objective(x, n):
    """
    Objective: minimize -sum(s_i)
    x: [x_0..x_n-1, y_0..y_n-1, s_0..s_n-1]
    """
    s = x[2*n:]
    return -np.sum(s)

def _nlp_gradient(x, n):
    """
    Gradient of objective.
    d/dx_i = 0, d/dy_i = 0, d/ds_i = -1
    """
    grad = np.zeros(3*n)
    grad[2*n:] = -1.0
    return grad

def _nlp_constraints(x, n, delta=1e-8):
    """
    Inequality constraints: g(x) >= 0
    1. Boundary constraints:
       x_i - s_i/2 >= 0
       1 - (x_i + s_i/2) >= 0
       y_i - s_i/2 >= 0
       1 - (y_i + s_i/2) >= 0
    2. Non-overlap constraints:
       (sqrt((x_i-x_j)^2+delta) - (s_i+s_j)/2) * (sqrt((y_i-y_j)^2+delta) - (s_i+s_j)/2) >= 0
    """
    cx = x[:n]
    cy = x[n:2*n]
    s = x[2*n:]
    
    cons = []
    
    # Boundary constraints
    for i in range(n):
        cons.append(cx[i] - s[i]/2)
        cons.append(1 - (cx[i] + s[i]/2))
        cons.append(cy[i] - s[i]/2)
        cons.append(1 - (cy[i] + s[i]/2))
        
    # Non-overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            # Smooth absolute value
            abs_dx = np.sqrt(dx*dx + delta)
            abs_dy = np.sqrt(dy*dy + delta)
            dist_x = abs_dx - (s[i] + s[j])/2
            dist_y = abs_dy - (s[i] + s[j])/2
            # Product must be >= 0
            cons.append(dist_x * dist_y)
            
    return np.array(cons)

def _nlp_jacobian(x, n, delta=1e-8):
    """
    Jacobian of constraints.
    This is complex to derive analytically for the product term with smooth abs.
    We will let SLSQP use numerical differentiation if we don't provide it, 
    or we can provide a simplified version.
    Given the small n=10, numerical differentiation might be acceptable but slow.
    Let's try to provide analytical gradients for the boundary constraints 
    and rely on numerical for the non-linear ones? No, SLSQP expects full Jacobian or None.
    If None, it uses finite differences.
    With 30 variables and ~50 constraints, finite differences is 30 * 50 = 1500 evaluations.
    SLSQP does this every iteration. Might be slow.
    
    Let's try to implement the Jacobian analytically.
    Let f(x) = g(x) * h(x) where g = |x_i-x_j| - (s_i+s_j)/2 and h = |y_i-y_j| - (s_i+s_j)/2.
    Using smooth abs: g = sqrt((x_i-x_j)^2+delta) - (s_i+s_j)/2.
    dg/dx_i = (x_i-x_j)/sqrt((x_i-x_j)^2+delta)
    dg/dx_j = -(x_i-x_j)/sqrt((x_i-x_j)^2+delta)
    dg/ds_i = -0.5
    dg/ds_j = -0.5
    
    Similarly for h with y.
    
    d(g*h)/dx = g' * h + g * h'
    
    We will build the Jacobian matrix.
    Constraints order:
    4n boundary constraints.
    n(n-1)/2 non-overlap constraints.
    """
    cx = x[:n]
    cy = x[n:2*n]
    s = x[2*n:]
    
    n_cons = 4*n + n*(n-1)//2
    J = np.zeros((n_cons, 3*n))
    
    # Boundary constraints gradients
    # c_0: x_i - s_i/2 >= 0
    # c_1: 1 - x_i - s_i/2 >= 0
    # c_2: y_i - s_i/2 >= 0
    # c_3: 1 - y_i - s_i/2 >= 0
    for i in range(n):
        row_idx = 4*i
        J[row_idx, i] = 1.0
        J[row_idx, 2*n+i] = -0.5
        
        row_idx = 4*i + 1
        J[row_idx, i] = -1.0
        J[row_idx, 2*n+i] = -0.5
        
        row_idx = 4*i + 2
        J[row_idx, n+i] = 1.0
        J[row_idx, 2*n+i] = -0.5
        
        row_idx = 4*i + 3
        J[row_idx, n+i] = -1.0
        J[row_idx, 2*n+i] = -0.5
        
    # Non-overlap constraints
    cons_idx = 4*n
    for i in range(n):
        for j in range(i+1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            
            denom_x = np.sqrt(dx*dx + delta)
            denom_y = np.sqrt(dy*dy + delta)
            
            g = denom_x - (s[i] + s[j])/2
            h = denom_y - (s[i] + s[j])/2
            
            # Derivatives of g
            dg_dx_i = dx / denom_x
            dg_dx_j = -dx / denom_x
            dg_ds_i = -0.5
            dg_ds_j = -0.5
            
            # Derivatives of h
            dh_dy_i = dy / denom_y
            dh_dy_j = -dy / denom_y
            dh_ds_i = -0.5
            dh_ds_j = -0.5
            
            # Gradient of g*h
            # d/dx_i = dg_dx_i * h
            # d/dx_j = dg_dx_j * h
            # d/dy_i = g * dh_dy_i
            # d/dy_j = g * dh_dy_j
            # d/ds_i = dg_ds_i * h + g * dh_ds_i
            # d/ds_j = dg_ds_j * h + g * dh_ds_j
            
            J[cons_idx, i] = dg_dx_i * h
            J[cons_idx, j] = dg_dx_j * h
            J[cons_idx, n+i] = g * dh_dy_i
            J[cons_idx, n+j] = g * dh_dy_j
            J[cons_idx, 2*n+i] = dg_ds_i * h + g * dh_ds_i
            J[cons_idx, 2*n+j] = dg_ds_j * h + g * dh_ds_j
            
            cons_idx += 1
            
    return J

def _get_grid_initial(n):
    k = int(np.ceil(np.sqrt(n)))
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    centers = np.stack([cx, cy], axis=1)
    sides = np.full(n, side)
    return np.concatenate([centers[:,0], centers[:,1], sides])

def _get_random_initial(n):
    # Random valid-ish packing
    # Place squares in a grid with some jitter and small sizes
    k = int(np.ceil(np.sqrt(n)))
    side = 0.8 / k # slightly smaller to allow jitter
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side + np.random.uniform(-0.1*side, 0.1*side, n)
    cy = (rows + 0.5) * side + np.random.uniform(-0.1*side, 0.1*side, n)
    # Clamp to [0,1]
    cx = np.clip(cx, 0, 1)
    cy = np.clip(cy, 0, 1)
    s = np.full(n, side)
    return np.concatenate([cx, cy, s])

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 630 - DEADLINE_MARGIN
    
    # Initial best from grid
    x_grid = _get_grid_initial(n)
    best_x = x_grid
    best_val = _nlp_objective(x_grid, n)
    
    # Generate multiple initial guesses
    initial_guesses = [x_grid]
    np.random.seed(42)
    for _ in range(5):
        initial_guesses.append(_get_random_initial(n))
        
    # Try a few structured LP-like guesses? 
    # We can just rely on SLSQP to move from grid.
    
    # Run SLSQP from each initial guess
    # To save time, we can run them in parallel? No, standard lib.
    # We will run them sequentially but with tight time limits.
    
    max_iter_per_run = 200
    time_per_run_limit = 30.0 # seconds
    
    for guess in initial_guesses:
        if time.time() > deadline:
            break
            
        run_start = time.time()
        
        try:
            # Bounds:
            # x, y in [0, 1]
            # s in [0, 1] (actually s <= 1)
            bounds = []
            for i in range(n):
                bounds.append((0, 1)) # x
            for i in range(n):
                bounds.append((0, 1)) # y
            for i in range(n):
                bounds.append((1e-6, 1)) # s
                
            # Constraints:
            # We pass the function that returns g(x) >= 0
            # SLSQP expects inequality constraints as g(x) >= 0
            
            res = minimize(
                _nlp_objective,
                guess,
                method='SLSQP',
                jac=_nlp_gradient,
                bounds=bounds,
                constraints=[
                    {'type': 'ineq', 'fun': _nlp_constraints, 'jac': _nlp_jacobian}
                ],
                options={'maxiter': max_iter_per_run, 'ftol': 1e-9, 'disp': False}
            )
            
            if res.success:
                x_res = res.x
                val = _nlp_objective(x_res, n)
                
                # Validate
                cx = x_res[:n]
                cy = x_res[n:2*n]
                s = x_res[2*n:]
                centers = np.stack([cx, cy], axis=1)
                if _is_valid(centers, s, n):
                    if val > best_val:
                        best_val = val
                        best_x = x_res.copy()
                        
        except Exception as e:
            pass
            
        # Check time for this run
        if time.time() - run_start > time_per_run_limit:
            # If one run takes too long, we might need to break or reduce iterations
            # But we have 600s total. 6 runs * 30s = 180s. Safe.
            pass
            
    # If we have time left, try more random restarts or refine the best
    remaining = deadline - time.time()
    if remaining > 60:
        # Refine the best solution with more iterations
        run_start = time.time()
        try:
            bounds = []
            for i in range(n):
                bounds.append((0, 1))
            for i in range(n):
                bounds.append((0, 1))
            for i in range(n):
                bounds.append((1e-6, 1))
                
            res = minimize(
                _nlp_objective,
                best_x,
                method='SLSQP',
                jac=_nlp_gradient,
                bounds=bounds,
                constraints=[
                    {'type': 'ineq', 'fun': _nlp_constraints, 'jac': _nlp_jacobian}
                ],
                options={'maxiter': 500, 'ftol': 1e-12, 'disp': False}
            )
            if res.success:
                x_res = res.x
                val = _nlp_objective(x_res, n)
                cx = x_res[:n]
                cy = x_res[n:2*n]
                s = x_res[2*n:]
                centers = np.stack([cx, cy], axis=1)
                if _is_valid(centers, s, n):
                    if val > best_val:
                        best_val = val
                        best_x = x_res.copy()
        except:
            pass

    # Extract final result
    cx = best_x[:n]
    cy = best_x[n:2*n]
    s = best_x[2*n:]
    centers = np.stack([cx, cy], axis=1)
    sides = s
    
    # Final safety check
    if not _is_valid(centers, sides, n):
        # Fallback to grid
        x_grid = _get_grid_initial(n)
        cx = x_grid[:n]
        cy = x_grid[n:2*n]
        s = x_grid[2*n:]
        centers = np.stack([cx, cy], axis=1)
        sides = s

    return centers, sides

# EVOLVE-BLOCK-END
