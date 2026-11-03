"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    best_score = -1
    best_result = None
    
    # Perimeter 4 => W + H = 2
    # Try different aspect ratios
    # Square (1x1) is often good for packing.
    # Also try slightly rectangular.
    
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.2]
    
    for w in aspect_ratios:
        if time.time() > deadline:
            break
        h = 2.0 - w
        if h <= 0:
            continue
        
        # Run optimization with a few different seeds
        for seed in range(3):
            if time.time() > deadline:
                break
            
            # Initial guess: Grid-like distribution
            # 5x5 grid = 25 spots, we need 21.
            # Let's just use random or a simple grid.
            # Random is safer for trust-constr to explore.
            rng = np.random.RandomState(seed)
            
            # x variables:
            # 0..N-1: cx
            # N..2N-1: cy
            # 2N..3N-1: r
            
            cx = rng.uniform(0.1, 0.9, N) * w
            cy = rng.uniform(0.1, 0.9, N) * h
            # Small initial radii to avoid immediate constraint violation
            r = np.full(N, 0.05 * min(w, h))
            
            x0 = np.concatenate([cx, cy, r])
            
            # Bounds
            # cx in [0, w], cy in [0, h], r in [0, min(w,h)/2]
            r_max = min(w, h)
            
            bounds = [(0.0, w)] * N + [(0.0, h)] * N + [(0.0, r_max)] * N
            
            # Objective: Maximize sum of radii => Minimize -sum(r)
            def objective(x):
                r = x[2*N:]
                return -np.sum(r)
            
            def objective_grad(x):
                g = np.zeros(3*N)
                g[2*N:] = -1.0
                return g
            
            # Constraints:
            # 1. Boundary: cx - r >= 0, cx + r <= w, cy - r >= 0, cy + r <= h
            # These are linear.
            # 2. Non-overlap: dist(i,j) >= r_i + r_j
            #    dist(i,j)^2 - (r_i + r_j)^2 >= 0
            #    This is a non-convex quadratic constraint.
            
            # We define constraints as g(x) >= 0.
            
            # To use trust-constr, we need:
            # fun, jac, hess (or hessp) for objective
            # fun, jac, hess (or hessp) for each constraint
            
            # Instead of defining 210 individual constraints (N*(N-1)/2),
            # which might be slow to set up, we can group them or just list them.
            # However, trust-constr expects a list of Constraint objects.
            # Creating 210 objects is fine.
            
            constraints = []
            
            # Boundary constraints
            # cx - r >= 0
            # cx + r - w >= 0
            # cy - r >= 0
            # cy + r - h >= 0
            
            # Let's add them one by one or as a batch?
            # Constraint objects take a single function.
            # It's more efficient to define the vector of all boundary violations?
            # No, scipy Constraint is for scalar constraints usually, or vector if jac returns vector?
            # Actually, scipy.optimize.minimize with trust-constr accepts:
            # constraints: sequence of Constraint objects.
            # Each Constraint has fun, jac, hess.
            # If fun returns a vector, jac must return a Jacobian matrix.
            
            # Let's define a single constraint function for ALL boundary constraints?
            # No, it's easier to just add 4*N constraints.
            
            for i in range(N):
                # cx_i - r_i >= 0
                def c_left(x, i=i):
                    return x[i] - x[2*N+i]
                def c_left_jac(x, i=i):
                    g = np.zeros(3*N)
                    g[i] = 1.0
                    g[2*N+i] = -1.0
                    return g
                constraints.append({'type': 'ineq', 'fun': c_left, 'jac': c_left_jac})
                
                # cx_i + r_i - w <= 0  =>  w - cx_i - r_i >= 0
                def c_right(x, i=i):
                    return w - x[i] - x[2*N+i]
                def c_right_jac(x, i=i):
                    g = np.zeros(3*N)
                    g[i] = -1.0
                    g[2*N+i] = -1.0
                    return g
                constraints.append({'type': 'ineq', 'fun': c_right, 'jac': c_right_jac})
                
                # cy_i - r_i >= 0
                def c_bottom(x, i=i):
                    return x[N+i] - x[2*N+i]
                def c_bottom_jac(x, i=i):
                    g = np.zeros(3*N)
                    g[N+i] = 1.0
                    g[2*N+i] = -1.0
                    return g
                constraints.append({'type': 'ineq', 'fun': c_bottom, 'jac': c_bottom_jac})
                
                # cy_i + r_i - h <= 0 => h - cy_i - r_i >= 0
                def c_top(x, i=i):
                    return h - x[N+i] - x[2*N+i]
                def c_top_jac(x, i=i):
                    g = np.zeros(3*N)
                    g[N+i] = -1.0
                    g[2*N+i] = -1.0
                    return g
                constraints.append({'type': 'ineq', 'fun': c_top, 'jac': c_top_jac})
            
            # Non-overlap constraints
            # dist(i,j)^2 - (r_i + r_j)^2 >= 0
            # (cx_i - cx_j)^2 + (cy_i - cy_j)^2 - (r_i + r_j)^2 >= 0
            
            pairs = []
            for i in range(N):
                for j in range(i+1, N):
                    pairs.append((i, j))
            
            for idx, (i, j) in enumerate(pairs):
                def c_ov(x, i=i, j=j):
                    dx = x[i] - x[j]
                    dy = x[N+i] - x[N+j]
                    ri = x[2*N+i]
                    rj = x[2*N+j]
                    return dx*dx + dy*dy - (ri + rj)**2
                
                def c_ov_jac(x, i=i, j=j):
                    g = np.zeros(3*N)
                    dx = x[i] - x[j]
                    dy = x[N+i] - x[N+j]
                    ri = x[2*N+i]
                    rj = x[2*N+j]
                    
                    # d/dx_i (dx^2) = 2*dx
                    # d/dx_j (dx^2) = -2*dx
                    # d/dy_i (dy^2) = 2*dy
                    # d/dy_j (dy^2) = -2*dy
                    # d/dr_i (-(ri+rj)^2) = -2*(ri+rj)
                    # d/dr_j (-(ri+rj)^2) = -2*(ri+rj)
                    
                    g[i] = 2 * dx
                    g[j] = -2 * dx
                    g[N+i] = 2 * dy
                    g[N+j] = -2 * dy
                    g[2*N+i] = -2 * (ri + rj)
                    g[2*N+j] = -2 * (ri + rj)
                    
                    return g
                
                constraints.append({'type': 'ineq', 'fun': c_ov, 'jac': c_ov_jac})
            
            try:
                res = minimize(
                    objective,
                    x0,
                    method='trust-constr',
                    jac=objective_grad,
                    hess='2-point', # Approximate Hessian
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 200, 'gtol': 1e-6, 'xtol': 1e-8}
                )
                
                x_opt = res.x
                # Validate
                cx_opt = x_opt[:N]
                cy_opt = x_opt[N:2*N]
                r_opt = x_opt[2*N:]
                
                # Check validity strictly
                valid = True
                # Bounds
                if np.any(cx_opt - r_opt < -1e-9) or np.any(cx_opt + r_opt - w > 1e-9):
                    valid = False
                if np.any(cy_opt - r_opt < -1e-9) or np.any(cy_opt + r_opt - h > 1e-9):
                    valid = False
                
                # Overlaps
                if valid:
                    for i in range(N):
                        for j in range(i+1, N):
                            dist = math.hypot(cx_opt[i] - cx_opt[j], cy_opt[i] - cy_opt[j])
                            if dist < r_opt[i] + r_opt[j] - 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    score = np.sum(r_opt)
                    if score > best_score:
                        best_score = score
                        best_result = np.column_stack((cx_opt, cy_opt, r_opt))
                        
            except Exception as e:
                # print(e)
                pass
            
            # If time is running out, break inner loop
            if time.time() > deadline - 5:
                break

    # Fallback: simple grid if optimization failed
    if best_result is None:
        # 5x5 grid in 1x1 square
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                r = 0.5 / side - 1e-4
                cells.append((cx, cy, r))
        best_result = np.array(cells[:N])
        
    return best_result
# EVOLVE-BLOCK-END
