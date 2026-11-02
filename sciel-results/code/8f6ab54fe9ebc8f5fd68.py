import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 32

def hex_init(n):
    """Generate a hexagonal lattice initialization for n circles."""
    # Hexagonal packing density is pi/sqrt(12) ~ 0.9069
    # For n=32, roughly 5x7 grid with some shifts
    # We want to fit n circles.
    # Simple approach: create a grid and take the first n, then optimize positions.
    # A better init: generate points on a hex lattice and pick the n closest to center?
    # Or just a square grid with hex offset.
    
    # Let's create a larger hex grid and select n points that fit best in [0,1]^2
    # Or simpler: 
    # Side length for hex lattice to fit ~sqrt(n) circles.
    # Let's try a grid of 6x6 = 36, take 32.
    # Or 5x7 = 35, take 32.
    
    # Let's use a heuristic grid that is slightly distorted
    # 6 columns, 6 rows
    cols = 6
    rows = 6
    points = []
    for i in range(cols):
        for j in range(rows):
            x = (i + 0.5) / cols
            y = (j + 0.5) / rows
            # Add hex offset
            if j % 2 == 1:
                x += 0.5 / cols
            # Clip to [0,1]
            x = np.clip(x, 0, 1)
            y = np.clip(y, 0, 1)
            points.append((x, y))
            
    # We have 36 points. We need 32.
    # To make it a valid starting point for optimization, we can just take the first 32.
    # But they might be clustered.
    # Better: randomize the selection or use the first N from a well-spread grid.
    
    # Let's try a different init: 
    # Place circles in a grid, but allow optimization to move them.
    # The optimizer will sort them out.
    
    # For the parameter vector:
    # We need r_i, u_i, v_i.
    # Initial r_i: assume equal radii. 
    # Area per circle ~ 1/N. Radius ~ sqrt(1/(4*pi*N)) ~ 0.05.
    # Let's start with r_i = 0.05.
    
    # Initial u_i, v_i: 
    # x_i = r_i + u_i(1-2r_i)
    # We want x_i to be roughly the grid point.
    # u_i = (x_grid - r_i) / (1 - 2r_i)
    
    r_init = np.full(n, 0.05)
    u_init = np.zeros(n)
    v_init = np.zeros(n)
    
    for i in range(n):
        if i < len(points):
            xg, yg = points[i]
        else:
            xg, yg = 0.5, 0.5 # Fallback
            
        # Ensure u, v are in [0,1] for the optimizer bounds
        denom = 1 - 2*r_init[i]
        if denom <= 0:
            u_init[i] = 0.5
            v_init[i] = 0.5
        else:
            u_init[i] = np.clip((xg - r_init[i]) / denom, 0, 1)
            v_init[i] = np.clip((yg - r_init[i]) / denom, 0, 1)
            
    return np.concatenate([r_init, u_init, v_init])

def objective_and_grad(params, lam):
    n = len(params) // 3
    r = params[0:n]
    u = params[n:2*n]
    v = params[2*n:3*n]
    
    # Compute x, y
    denom = 1 - 2*r
    # Handle denom <= 0 (r >= 0.5). If r >= 0.5, only 1 circle fits. 
    # For n=32, r should be small.
    # Clip denom to avoid division by zero
    denom_safe = np.where(denom > 1e-6, denom, 1e-6)
    x = r + u * denom_safe
    y = r + v * denom_safe
    
    # Objective: -sum(r) + penalty
    obj = -np.sum(r)
    grad = np.zeros_like(params)
    
    # Gradient of -sum(r)
    grad[0:n] -= 1.0
    
    # Penalty term
    # Sum over pairs i < j of max(0, r_i + r_j - d_ij)^2
    # d_ij = sqrt((x_i-x_j)^2 + (y_i-y_j)^2)
    
    # To be efficient, we can loop, but n=32 is small (496 pairs).
    # Vectorized calculation is possible but loop is fine for clarity and gradient correctness.
    
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d2 = dx*dx + dy*dy
            d = np.sqrt(d2 + 1e-12) # Avoid 0 division
            r_sum = r[i] + r[j]
            
            if d < r_sum:
                violation = r_sum - d
                # Penalty: violation^2
                # Gradient wrt params
                # d(d)/d(x_i) = dx/d, d(x_i)/d(r_i) = 1 - 2u_i, d(x_i)/d(u_i) = 1-2r_i
                # d(d)/d(r_i) = 0 (d depends on x,y, which depend on r, but in the term r_sum - d, the r_sum part is explicit)
                # Wait, x_i depends on r_i. So d depends on r_i.
                # d = sqrt((x_i-x_j)^2 + (y_i-y_j)^2)
                # x_i = r_i + u_i(1-2r_i)
                # dx_i/dr_i = 1 - 2u_i
                # dx_i/du_i = 1 - 2r_i
                # dx_i/dv_i = 0
                
                # Let's compute derivatives of d
                inv_d = 1.0 / d
                
                # dd_dx_i = dx * inv_d
                # dd_dy_i = dy * inv_d
                # dd_dr_i = dd_dx_i * dx_i_dr_i + dd_dy_i * dy_i_dr_i
                # dx_i_dr_i = 1 - 2*u[i]
                # dy_i_dr_i = 1 - 2*v[i]
                # dd_dr_i = (dx * inv_d) * (1 - 2*u[i]) + (dy * inv_d) * (1 - 2*v[i])
                
                # dd_dr_j = -dd_dr_i (by symmetry of distance, but careful with signs)
                # Actually, d = ||x_i - x_j||. 
                # d/dr_j = (x_j - x_i)/d * dx_j/dr_j + ... = - (x_i - x_j)/d * dx_j/dr_j ...
                # Let's just compute for i and j separately to be safe.
                
                dx_dr_i = 1.0 - 2.0 * u[i]
                dy_dr_i = 1.0 - 2.0 * v[i]
                dx_du_i = 1.0 - 2.0 * r[i]
                dy_dv_i = 1.0 - 2.0 * r[i]
                
                dd_dr_i = (dx * inv_d) * dx_dr_i + (dy * inv_d) * dy_dr_i
                dd_du_i = (dx * inv_d) * dx_du_i
                dd_dv_i = (dy * inv_d) * dy_dv_i
                
                # For j:
                # x_j - x_i = -dx
                # dx_j/dr_j = 1 - 2*u[j]
                # dd/dr_j = (-dx * inv_d) * (1 - 2*u[j]) + (-dy * inv_d) * (1 - 2*v[j])
                # Note: dx = x_i - x_j.
                # d/dx_j (x_i - x_j)^2 = -2(x_i - x_j)
                # So derivative of d wrt x_j is -(x_i - x_j)/d = -dx/d.
                
                dd_dr_j = (-dx * inv_d) * (1.0 - 2.0 * u[j]) + (-dy * inv_d) * (1.0 - 2.0 * v[j])
                dd_du_j = (-dx * inv_d) * (1.0 - 2.0 * r[j])
                dd_dv_j = (-dy * inv_d) * (1.0 - 2.0 * r[j])
                
                # Gradient of penalty term P = (r_sum - d)^2
                # dP/dr_i = 2(r_sum - d) * (1 - dd_dr_i)
                # dP/du_i = 2(r_sum - d) * (0 - dd_du_i) = -2(r_sum - d) * dd_du_i
                # dP/dv_i = -2(r_sum - d) * dd_dv_i
                
                factor = 2.0 * violation * lam
                
                grad[i] += factor * (1.0 - dd_dr_i)
                grad[n+i] += factor * (-dd_du_i)
                grad[2*n+i] += factor * (-dd_dv_i)
                
                grad[j] += factor * (1.0 - dd_dr_j)
                grad[n+j] += factor * (-dd_du_j)
                grad[2*n+j] += factor * (-dd_dv_j)
                
                obj += lam * violation * violation
                
    return obj, grad

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin
    
    best_score = -np.inf
    best_params = None
    
    # Multiple restarts
    n_restarts = 20
    
    for k in range(n_restarts):
        if time.time() > deadline:
            break
            
        # Generate init
        if k == 0:
            params = hex_init(N)
        else:
            # Perturb the hex init
            params = hex_init(N)
            noise = np.random.normal(0, 0.05, params.shape)
            params = params + noise
            # Ensure bounds roughly
            # r > 0
            params[0:N] = np.maximum(params[0:N], 0.01)
            # u, v in [0,1]
            params[N:2*N] = np.clip(params[N:2*N], 0, 1)
            params[2*N:3*N] = np.clip(params[2*N:3*N], 0, 1)
            
        # Optimize
        # Use L-BFGS-B
        # Bounds: r in [0.01, 0.5], u,v in [0,1]
        bounds = [(0.01, 0.5)] * N + [(0, 1)] * (2*N)
        
        # Start with lower lambda, increase if needed? 
        # Or just use a fixed high lambda.
        # Let's try lambda = 100.
        lam = 100.0
        
        try:
            res = minimize(
                lambda p: objective_and_grad(p, lam)[0],
                params,
                jac=lambda p: objective_and_grad(p, lam)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < best_score:
                # Check validity
                r = res.x[0:N]
                u = res.x[N:2*N]
                v = res.x[2*N:3*N]
                
                # Verify no overlaps and inside square
                valid = True
                x = r + u * (1 - 2*r)
                y = r + v * (1 - 2*r)
                
                for i in range(N):
                    if x[i] < r[i] - 1e-6 or x[i] > 1 - r[i] + 1e-6 or y[i] < r[i] - 1e-6 or y[i] > 1 - r[i] + 1e-6:
                        valid = False
                        break
                    if r[i] <= 0:
                        valid = False
                        break
                        
                if valid:
                    for i in range(N):
                        for j in range(i+1, N):
                            d2 = (x[i]-x[j])**2 + (y[i]-y[j])**2
                            if d2 < (r[i]+r[j])**2 - 1e-6:
                                valid = False
                                break
                        if not valid: break
                
                if valid:
                    score = np.sum(r)
                    if score > best_score:
                        best_score = score
                        best_params = res.x.copy()
                        
        except Exception as e:
            pass

    # If no valid solution found, return a trivial one
    if best_params is None:
        # Trivial grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Format output
    r = best_params[0:N]
    u = best_params[N:2*N]
    v = best_params[2*N:3*N]
    x = r + u * (1 - 2*r)
    y = r + v * (1 - 2*r)
    
    return np.column_stack((x, y, r))
