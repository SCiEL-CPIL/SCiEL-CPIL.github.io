"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _compute_objective(params, n):
    """
    Objective: maximize sum of radii.
    We minimize negative sum of radii.
    params: array of shape (3n,) = [x1,y1,r1, x2,y2,r2, ..., xn,yn,rn]
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    # Boundary constraints: circles must be inside [0,1]^2
    # x_i - r_i >= 0, x_i + r_i <= 1
    # y_i - r_i >= 0, y_i + r_i <= 1
    # We use a soft penalty for boundary violations to keep it differentiable
    # Or we can use SLSQP with constraints. Let's try SLSQP with constraints first.
    
    # Actually, for SLSQP, it's better to handle constraints explicitly.
    # But the objective is just -sum(r).
    # Constraints:
    # 1. r_i >= 0
    # 2. x_i - r_i >= 0
    # 3. x_i + r_i <= 1
    # 4. y_i - r_i >= 0
    # 5. y_i + r_i <= 1
    # 6. (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2 for all i < j
    
    # This is a non-convex problem.
    # Let's use a penalty method or a trust region method.
    # SLSQP can handle non-linear constraints.
    
    # To speed up, we can compute the objective simply as -sum(r).
    return -np.sum(r)

def _constraints(params, n):
    """
    List of constraint dictionaries for SLSQP.
    Each constraint: {'type': 'ineq', 'fun': lambda x: ...}
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    cons = []
    
    # 1. r_i >= 0
    # This is linear, can be handled by bounds, but let's add it just in case or use bounds.
    # SLSQP supports bounds. We'll use bounds for r >= 0, x,y in [0,1].
    # But r can be up to 0.5.
    
    # 2. x_i - r_i >= 0  =>  r_i - x_i <= 0
    # 3. x_i + r_i <= 1  =>  x_i + r_i - 1 <= 0
    # 4. y_i - r_i >= 0  =>  r_i - y_i <= 0
    # 5. y_i + r_i <= 1  =>  y_i + r_i - 1 <= 0
    
    # It's more efficient to pass these as array-valued constraints if possible, 
    # but SLSQP expects individual constraints or we can group them.
    # Let's group boundary constraints into one vector-valued constraint.
    
    # Boundary constraints (ineq, must be >= 0):
    # x_i - r_i >= 0
    # 1 - (x_i + r_i) >= 0
    # y_i - r_i >= 0
    # 1 - (y_i + r_i) >= 0
    
    # Let's define a function that returns an array of these values.
    # SLSQP allows 'fun' to return an array for a single constraint dict?
    # No, standard SLSQP expects scalar or we can use multiple dicts.
    # Actually, scipy.optimize.minimize with method='SLSQP' accepts a list of constraint dicts.
    # If 'fun' returns an array, it's treated as multiple constraints.
    
    def boundary_constraints(params):
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        return np.concatenate([
            x - r,      # x_i - r_i >= 0
            1 - (x + r), # 1 - x_i - r_i >= 0
            y - r,      # y_i - r_i >= 0
            1 - (y + r)  # 1 - y_i - r_i >= 0
        ])
    
    cons.append({'type': 'ineq', 'fun': boundary_constraints})
    
    # Pairwise non-overlap constraints:
    # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    
    # This is O(n^2) constraints. For n=32, that's 496 constraints.
    # SLSQP might struggle with so many non-linear constraints.
    # Alternative: Use a penalty method with a smooth approximation of the non-overlap constraint.
    # Or use a different solver like 'trust-constr' which handles many constraints better.
    
    # Let's try 'trust-constr' instead. It's more robust for large numbers of constraints.
    
    return cons

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # 140s out of 150s limit
    
    # Initial guess: Grid
    side = int(np.ceil(np.sqrt(N)))
    x0 = np.zeros(N)
    y0 = np.zeros(N)
    r0 = np.zeros(N)
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= N:
                break
            x0[idx] = (i + 0.5) / side
            y0[idx] = (j + 0.5) / side
            r0[idx] = 0.5 / side
            idx += 1
    if idx < N:
        # Fill remaining (should not happen if side*side >= N)
        for k in range(idx, N):
            x0[k] = 0.5
            y0[k] = 0.5
            r0[k] = 0.1
            
    params0 = np.zeros(3 * N)
    params0[0::3] = x0
    params0[1::3] = y0
    params0[2::3] = r0
    
    # Bounds
    # x, y in [0, 1]
    # r in [0, 0.5]
    bounds = []
    for i in range(N):
        bounds.append((0, 1))  # x
        bounds.append((0, 1))  # y
        bounds.append((0, 0.5))  # r
    
    # Objective function
    def objective(params):
        r = params[2::3]
        return -np.sum(r)
    
    # Constraints for trust-constr
    # Non-overlap: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    # => (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    
    # Boundary constraints can be handled by bounds, but r depends on x,y.
    # x - r >= 0 => x - params[2+i] >= 0
    # This is a non-linear constraint because r is a variable.
    # So we must include boundary constraints in the constraint set.
    
    # Let's define the constraints for trust-constr.
    # It accepts LinearConstraint and NonlinearConstraint.
    # Boundary constraints are non-linear (because of r).
    
    def boundary_func(params):
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        return np.concatenate([
            x - r,
            1 - (x + r),
            y - r,
            1 - (y + r)
        ])
    
    def boundary_jac(params):
        # Jacobian of boundary_func w.r.t. params
        # Shape: (4N, 3N)
        # For each i:
        # d/dx_i (x_i - r_i) = 1
        # d/dr_i (x_i - r_i) = -1
        # d/dx_i (1 - x_i - r_i) = -1
        # d/dr_i (1 - x_i - r_i) = -1
        # etc.
        
        jac = np.zeros((4 * N, 3 * N))
        for i in range(N):
            # x_i - r_i
            jac[i, i] = 1.0       # d/dx_i
            jac[i, 2 * i + 2] = -1.0 # d/dr_i
            
            # 1 - x_i - r_i
            jac[N + i, i] = -1.0
            jac[N + i, 2 * i + 2] = -1.0
            
            # y_i - r_i
            jac[2 * N + i, i + N] = 1.0  # d/dy_i
            jac[2 * N + i, 2 * i + 2] = -1.0
            
            # 1 - y_i - r_i
            jac[3 * N + i, i + N] = -1.0
            jac[3 * N + i, 2 * i + 2] = -1.0
            
        return jac
    
    # Non-overlap constraints
    # There are N*(N-1)/2 constraints.
    # Let's compute them.
    def nonoverlap_func(params):
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        vals = []
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dr = r[i] + r[j]
                vals.append(dx * dx + dy * dy - dr * dr)
        return np.array(vals)
    
    def nonoverlap_jac(params):
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        m = N * (N - 1) // 2
        jac = np.zeros((m, 3 * N))
        k = 0
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dr = r[i] + r[j]
                
                # d/dx_i (dx^2 + dy^2 - dr^2) = 2 * dx
                # d/dx_j = -2 * dx
                # d/dy_i = 2 * dy
                # d/dy_j = -2 * dy
                # d/dr_i = -2 * dr
                # d/dr_j = -2 * dr
                
                jac[k, i] = 2 * dx
                jac[k, j] = -2 * dx
                jac[k, i + N] = 2 * dy
                jac[k, j + N] = -2 * dy
                jac[k, 2 * i + 2] = -2 * dr
                jac[k, 2 * j + 2] = -2 * dr
                k += 1
        return jac
    
    from scipy.optimize import NonlinearConstraint, LinearConstraint, trust_constr
    
    # Boundary constraints: >= 0
    nl_con_boundary = NonlinearConstraint(boundary_func, 0, np.inf, jac=boundary_jac)
    
    # Non-overlap constraints: >= 0
    nl_con_nonoverlap = NonlinearConstraint(nonoverlap_func, 0, np.inf, jac=nonoverlap_jac)
    
    constraints = [nl_con_boundary, nl_con_nonoverlap]
    
    best_params = params0.copy()
    best_obj = objective(params0)
    
    # Try multiple random restarts?
    # Grid is a good start. Let's try to optimize from grid.
    # If it fails, try random starts.
    
    try:
        res = minimize(
            objective,
            params0,
            method='trust-constr',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'gtol': 1e-6, 'xtol': 1e-6}
        )
        
        if res.success or res.fun < best_obj:
            # Check validity
            x = res.x[0::3]
            y = res.x[1::3]
            r = res.x[2::3]
            
            # Verify constraints numerically
            valid = True
            if np.any(r < -1e-6) or np.any(x < -1e-6) or np.any(y < -1e-6) or np.any(x > 1 + 1e-6) or np.any(y > 1 + 1e-6):
                valid = False
            if np.any(x - r < -1e-5) or np.any(1 - (x + r) < -1e-5) or np.any(y - r < -1e-5) or np.any(1 - (y + r) < -1e-5):
                valid = False
            for i in range(N):
                for j in range(i + 1, N):
                    dist2 = (x[i] - x[j])**2 + (y[i] - y[j])**2
                    sum_r2 = (r[i] + r[j])**2
                    if dist2 < sum_r2 - 1e-5:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Clip tiny negative values
                res.x[2::3] = np.maximum(res.x[2::3], 0)
                res.x[0::3] = np.clip(res.x[0::3], 0, 1)
                res.x[1::3] = np.clip(res.x[1::3], 0, 1)
                best_params = res.x
                best_obj = res.fun
                
    except Exception as e:
        pass
    
    # If time remains, try a few random restarts
    if time.time() < deadline - 10:
        np.random.seed(42)
        for restart in range(10):
            if time.time() > deadline - 5:
                break
                
            # Random initial guess
            params_rand = np.random.rand(3 * N)
            params_rand[0::3] *= 0.8 + 0.1 * np.random.rand(N)  # x in [0.1, 0.9] roughly
            params_rand[1::3] *= 0.8 + 0.1 * np.random.rand(N)  # y
            params_rand[2::3] *= 0.2  # r small initially
            
            try:
                res = minimize(
                    objective,
                    params_rand,
                    method='trust-constr',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 300, 'gtol': 1e-6, 'xtol': 1e-6}
                )
                
                if res.fun < best_obj:
                    x = res.x[0::3]
                    y = res.x[1::3]
                    r = res.x[2::3]
                    
                    valid = True
                    if np.any(r < -1e-6) or np.any(x < -1e-6) or np.any(y < -1e-6) or np.any(x > 1 + 1e-6) or np.any(y > 1 + 1e-6):
                        valid = False
                    if np.any(x - r < -1e-5) or np.any(1 - (x + r) < -1e-5) or np.any(y - r < -1e-5) or np.any(1 - (y + r) < -1e-5):
                        valid = False
                    for i in range(N):
                        for j in range(i + 1, N):
                            dist2 = (x[i] - x[j])**2 + (y[i] - y[j])**2
                            sum_r2 = (r[i] + r[j])**2
                            if dist2 < sum_r2 - 1e-5:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        res.x[2::3] = np.maximum(res.x[2::3], 0)
                        res.x[0::3] = np.clip(res.x[0::3], 0, 1)
                        res.x[1::3] = np.clip(res.x[1::3], 0, 1)
                        best_params = res.x
                        best_obj = res.fun
                        
            except Exception as e:
                pass
    
    # Construct output array
    x = best_params[0::3]
    y = best_params[1::3]
    r = best_params[2::3]
    
    # Final safety clip
    r = np.maximum(r, 0)
    x = np.clip(x, 0, 1)
    y = np.clip(y, 0, 1)
    
    # Ensure boundary constraints are met by shrinking if necessary
    for i in range(N):
        max_r = min(x[i], 1 - x[i], y[i], 1 - y[i])
        if r[i] > max_r:
            r[i] = max_r * 0.99
    
    # Ensure non-overlap by shrinking if necessary (crude)
    # This is a last resort, might reduce score but ensures validity
    for iter in range(10):
        adjusted = False
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                sum_r = r[i] + r[j]
                if dist < sum_r:
                    # Scale down both radii
                    scale = dist / (sum_r * 1.01) if sum_r > 0 else 0
                    r[i] *= scale
                    r[j] *= scale
                    adjusted = True
        if not adjusted:
            break
    
    # Final boundary check
    for i in range(N):
        max_r = min(x[i], 1 - x[i], y[i], 1 - y[i])
        if r[i] > max_r:
            r[i] = max_r * 0.99
    
    result = np.column_stack((x, y, r))
    return result

# EVOLVE-BLOCK-END
