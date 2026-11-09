import numpy as np
import time
from scipy.optimize import minimize, linprog

N = 26
_DEADLINE = 140.0  # Safety margin from 150s limit

def is_valid(x, y, r, tol=1e-6):
    """
    Checks if the configuration is valid:
    1. All radii are non-negative.
    2. All circles are inside the unit square.
    3. No circles overlap.
    """
    if np.any(r < -tol):
        return False
    if np.any(x < r - tol) or np.any(x > 1 - r + tol):
        return False
    if np.any(y < r - tol) or np.any(y > 1 - r + tol):
        return False
    
    # Check pairwise distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    min_dist = r[:, None] + r[None, :]
    
    # Check for overlaps with tolerance
    if np.any(dist < min_dist - tol):
        return False
    return True

def solve_radii_lp(x, y, epsilon=1e-9):
    """
    Solves the Linear Programming problem to maximize sum(r)
    given fixed centers x, y.
    """
    n = len(x)
    c = -np.ones(n)
    
    A_rows = []
    b_vals = []
    
    # Pairwise constraints: r_i + r_j <= d_ij - epsilon
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            if d < epsilon:
                return None
            row = np.zeros(n)
            row[i] = 1
            row[j] = 1
            A_rows.append(row)
            b_vals.append(d - epsilon)
            
    # Boundary constraints: r_i <= min(x_i, 1-x_i, y_i, 1-y_i) - epsilon
    b_i = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
    for i in range(n):
        row = np.zeros(n)
        row[i] = 1
        A_rows.append(row)
        b_vals.append(b_i[i] - epsilon)
        
    A_ub = np.array(A_rows)
    b_ub = np.array(b_vals)
    bounds = [(0, None)] * n
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            r = res.x
            # Verify validity
            if is_valid(x, y, r):
                return r
    except Exception:
        pass
        
    return None

def objective_and_grad(z, penalty_weight):
    """
    Objective function and gradient for L-BFGS-B.
    z is a vector of length 3*N: [x_0, ..., x_N-1, y_0, ..., y_N-1, r_0, ..., r_N-1]
    We want to maximize sum(r), so we minimize -sum(r) + penalty(violations).
    
    Penalty is quadratic in violations to ensure smoothness for gradient-based methods.
    Violations:
    1. Overlap: max(0, r_i + r_j - dist_ij)
    2. Boundary: max(0, r_i - x_i), max(0, r_i - (1-x_i)), etc.
    """
    N = len(z) // 3
    x = z[:N]
    y = z[N:2*N]
    r = z[2*N:]
    
    # We want to maximize sum(r), so objective is -sum(r)
    obj = -np.sum(r)
    grad_x = np.zeros(N)
    grad_y = np.zeros(N)
    grad_r = -np.ones(N)
    
    # --- Overlap Penalty ---
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    
    # Avoid division by zero in gradient calculation
    dist_safe = np.where(dist > 1e-6, dist, 1e-6)
    
    min_dist = r[:, None] + r[None, :]
    overlap = np.maximum(0, min_dist - dist)
    np.fill_diagonal(overlap, 0)
    
    # Penalty term: penalty_weight * sum(overlap^2)
    obj += penalty_weight * np.sum(overlap**2)
    
    # Gradient of penalty w.r.t x_i
    # d/dx_i (overlap_ij^2) = 2 * overlap_ij * d(overlap_ij)/dx_i
    # overlap_ij = r_i + r_j - dist_ij
    # d(dist_ij)/dx_i = (x_i - x_j) / dist_ij = ux_ij
    # d(overlap_ij)/dx_i = -ux_ij
    # So grad_x_i += penalty_weight * 2 * overlap_ij * (-ux_ij)
    # But we are minimizing, so we add the gradient of the penalty to the gradient of the objective.
    # Wait, standard gradient descent: z_new = z - lr * grad(f).
    # f = -sum(r) + penalty.
    # grad_x(f) = penalty_weight * sum_j 2 * overlap_ij * (-ux_ij)
    # So we add this to grad_x.
    
    ux = dx / dist_safe
    uy = dy / dist_safe
    
    grad_x += -2.0 * penalty_weight * np.sum(overlap * ux, axis=1)
    grad_y += -2.0 * penalty_weight * np.sum(overlap * uy, axis=1)
    
    # Gradient w.r.t r_i
    # d(overlap_ij)/dr_i = 1
    # grad_r_i += penalty_weight * sum_j 2 * overlap_ij * 1
    grad_r += 2.0 * penalty_weight * np.sum(overlap, axis=1)
    
    # --- Boundary Penalty ---
    # Violations:
    # vx1 = max(0, r - x)  -> if x < r
    # vx2 = max(0, r - (1-x)) -> if x > 1-r
    # vy1 = max(0, r - y)
    # vy2 = max(0, r - (1-y))
    
    vx1 = np.maximum(0, r - x)
    vx2 = np.maximum(0, r - (1 - x))
    vy1 = np.maximum(0, r - y)
    vy2 = np.maximum(0, r - (1 - y))
    
    bound_viol = vx1**2 + vx2**2 + vy1**2 + vy2**2
    obj += penalty_weight * np.sum(bound_viol)
    
    # Gradients
    # d/dx (vx1^2) = 2*vx1 * (-1) = -2*vx1
    # d/dx (vx2^2) = 2*vx2 * (1) = 2*vx2
    grad_x += penalty_weight * (-2.0 * vx1 + 2.0 * vx2)
    
    # d/dy (vy1^2) = -2*vy1
    # d/dy (vy2^2) = 2*vy2
    grad_y += penalty_weight * (-2.0 * vy1 + 2.0 * vy2)
    
    # d/dr (vx1^2) = 2*vx1
    # d/dr (vx2^2) = 2*vx2
    # ...
    grad_r += penalty_weight * (2.0 * vx1 + 2.0 * vx2 + 2.0 * vy1 + 2.0 * vy2)
    
    # Combine gradients
    grad = np.concatenate([grad_x, grad_y, grad_r])
    
    return obj, grad

def l_bfgsb_optimize(x, y, r, start_time, deadline):
    """
    Runs L-BFGS-B optimization starting from initial state x, y, r.
    Returns best score and config found during the run.
    """
    best_score = 0.0
    best_config = None
    
    # Initial state
    z0 = np.concatenate([x, y, r])
    
    # Bounds:
    # x, y in [0, 1]
    # r in [0, 0.5] (max possible radius in unit square)
    bounds = [(0, 1)] * N + [(0, 1)] * N + [(0, 0.5)] * N
    
    # Penalty weight starts high to enforce constraints, then we might lower it?
    # Actually, for packing, we want to maximize radii.
    # If we start with small radii, there is no overlap, so penalty is 0.
    # As radii grow, overlaps appear, penalty pushes them apart or shrinks them.
    # A constant high penalty weight usually works well to keep them valid.
    # Let's try a high weight.
    penalty_weight = 100.0
    
    # Callback to track best valid configuration
    def callback(z):
        nonlocal best_score, best_config
        nonlocal penalty_weight
        
        x_c = z[:N]
        y_c = z[N:2*N]
        r_c = z[2*N:]
        
        # Check validity
        if is_valid(x_c, y_c, r_c):
            score = np.sum(r_c)
            if score > best_score:
                best_score = score
                best_config = (x_c.copy(), y_c.copy(), r_c.copy())
                
        # Adaptive penalty: if we are making progress in radius sum, we can slightly lower penalty?
        # Or keep it constant. Let's keep it constant for stability in L-BFGS-B.
        # However, if we are stuck, increasing penalty might help escape?
        # L-BFGS-B handles the quadratic penalty well.
        
        # Check time
        if time.time() > deadline - 5:
            # We can't easily stop L-BFGS-B from callback, but we can signal?
            # Actually, scipy minimize doesn't support early stopping via callback easily.
            # We rely on maxiter and time checks in the outer loop.
            pass
        
    try:
        res = minimize(
            objective_and_grad,
            z0,
            args=(penalty_weight,),
            jac=True,  # We return (f, grad)
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
        )
        
        # Check final result
        z_final = res.x
        x_f = z_final[:N]
        y_f = z_final[N:2*N]
        r_f = z_final[2*N:]
        
        if is_valid(x_f, y_f, r_f):
            score = np.sum(r_f)
            if score > best_score:
                best_score = score
                best_config = (x_f.copy(), y_f.copy(), r_f.copy())
                
    except Exception as e:
        pass
        
    return best_score, best_config

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    best_score = 0.0
    best_config = None
    
    n_restarts = 50  # More restarts are possible with L-BFGS-B if it's fast
    
    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            break
            
        # Initialization strategies
        if restart % 5 == 0:
            # Grid init
            side = int(np.ceil(np.sqrt(N)))
            xs = []
            ys = []
            for i in range(side):
                for j in range(side):
                    if len(xs) < N:
                        xs.append((i + 0.5) / side)
                        ys.append((j + 0.5) / side)
            x = np.array(xs[:N])
            y = np.array(ys[:N])
            r = np.full(N, 0.02) # Small initial radii
            
        elif restart % 5 == 1:
            # Random init
            x = np.random.uniform(0.1, 0.9, N)
            y = np.random.uniform(0.1, 0.9, N)
            r = np.full(N, 0.02)
            
        elif restart % 5 == 2:
            # Perturbed grid
            side = int(np.ceil(np.sqrt(N)))
            xs = []
            ys = []
            for i in range(side):
                for j in range(side):
                    if len(xs) < N:
                        xs.append((i + 0.5) / side)
                        ys.append((j + 0.5) / side)
            x = np.array(xs[:N])
            y = np.array(ys[:N])
            x += np.random.normal(0, 0.05, N)
            y += np.random.normal(0, 0.05, N)
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            r = np.full(N, 0.02)
            
        elif restart % 5 == 3:
            # Hexagonal-like init
            rows = 5
            cols = 6
            xs = []
            ys = []
            for i in range(rows):
                for j in range(cols):
                    if len(xs) < N:
                        xs.append((j + 0.5 + (0.5 if i % 2 else 0)) / cols)
                        ys.append((i + 0.5) / rows)
            x = np.array(xs[:N])
            y = np.array(ys[:N])
            x += np.random.normal(0, 0.02, N)
            y += np.random.normal(0, 0.02, N)
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            r = np.full(N, 0.02)
            
        else:
            # Random with varied radii
            x = np.random.uniform(0.1, 0.9, N)
            y = np.random.uniform(0.1, 0.9, N)
            r = np.random.uniform(0.01, 0.05, N)
            
        # Run L-BFGS-B optimization
        score, config = l_bfgsb_optimize(x, y, r, start_time, deadline)
        
        if config is not None:
            if score > best_score:
                best_score = score
                best_config = config
                
        # If we have a good config, try LP refinement on the best one found so far
        if best_config is not None and restart % 10 == 0:
            bx, by, br = best_config
            r_lp = solve_radii_lp(bx, by)
            if r_lp is not None:
                score_lp = np.sum(r_lp)
                if score_lp > best_score:
                    best_score = score_lp
                    best_config = (bx.copy(), by.copy(), r_lp.copy())
                    
        # Also try LP refinement on the current restart's result if it's good
        if config is not None and score > 0.8 * best_score:
            bx, by, br = config
            r_lp = solve_radii_lp(bx, by)
            if r_lp is not None:
                score_lp = np.sum(r_lp)
                if score_lp > best_score:
                    best_score = score_lp
                    best_config = (bx.copy(), by.copy(), r_lp.copy())

    # Fallback
    if best_config is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    bx, by, br = best_config
    return np.column_stack((bx, by, br))
