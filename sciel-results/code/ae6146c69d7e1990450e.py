"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _objective_and_grad(params):
    """
    params: array of shape (3N,) = [x1,y1,r1, ..., xN,yN,rN]
    We want to maximize sum(r_i) subject to:
      - circles disjoint: dist(i,j) >= r_i + r_j
      - inside unit square: r_i <= x_i <= 1-r_i, r_i <= y_i <= 1-r_i
      - r_i > 0
    
    We use a barrier/augmented penalty approach.
    Objective to minimize: -sum(r_i) + penalty
    """
    n = N
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    # Penalty for violations
    pen = 0.0
    grad_pen = np.zeros_like(params)
    
    # Wall constraints: x >= r, x <= 1-r, y >= r, y <= 1-r
    # Use quadratic penalty
    mu_wall = 100.0
    
    # x >= r => x - r >= 0
    viol_x_left = np.maximum(0, r - x)
    viol_x_right = np.maximum(0, x + r - 1)
    viol_y_left = np.maximum(0, r - y)
    viol_y_right = np.maximum(0, y + r - 1)
    
    pen += mu_wall * (np.sum(viol_x_left**2) + np.sum(viol_x_right**2) + 
                      np.sum(viol_y_left**2) + np.sum(viol_y_right**2))
    
    # Gradients for wall penalties
    # d/dx of mu*(r-x)^2 for x<r: -2*mu*(r-x) = 2*mu*(x-r)
    # d/dx of mu*(x+r-1)^2 for x>1-r: 2*mu*(x+r-1)
    # d/dr of mu*(r-x)^2 for x<r: 2*mu*(r-x)
    # d/dr of mu*(x+r-1)^2 for x>1-r: 2*mu*(x+r-1)
    
    grad_x = np.zeros(n)
    grad_y = np.zeros(n)
    grad_r = np.zeros(n)
    
    mask_xl = viol_x_left > 0
    mask_xr = viol_x_right > 0
    mask_yl = viol_y_left > 0
    mask_yr = viol_y_right > 0
    
    grad_x[mask_xl] += -2 * mu_wall * viol_x_left[mask_xl]
    grad_r[mask_xl] += 2 * mu_wall * viol_x_left[mask_xl]
    
    grad_x[mask_xr] += 2 * mu_wall * viol_x_right[mask_xr]
    grad_r[mask_xr] += 2 * mu_wall * viol_x_right[mask_xr]
    
    grad_y[mask_yl] += -2 * mu_wall * viol_y_left[mask_yl]
    grad_r[mask_yl] += 2 * mu_wall * viol_y_left[mask_yl]
    
    grad_y[mask_yr] += 2 * mu_wall * viol_y_right[mask_yr]
    grad_r[mask_yr] += 2 * mu_wall * viol_y_right[mask_yr]
    
    # Circle-circle constraints: dist(i,j) >= r_i + r_j
    mu_cc = 100.0
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = r[i] + r[j]
            if dist < 1e-10:
                dist = 1e-10
            viol = max(0, min_dist - dist)
            if viol > 0:
                pen += mu_cc * viol**2
                # d/dx_i of mu*(min_dist - dist)^2 = 2*mu*(min_dist-dist) * d(dist)/d(x_i)
                # d(dist)/d(x_i) = (x_i - x_j)/dist
                factor = 2 * mu_cc * viol
                grad_x[i] += factor * (dx / dist)
                grad_y[i] += factor * (dy / dist)
                grad_x[j] += factor * (-dx / dist)
                grad_y[j] += factor * (-dy / dist)
                grad_r[i] += factor
                grad_r[j] += factor
    
    # Objective: -sum(r) + pen
    obj = -np.sum(r) + pen
    
    # Gradient of objective
    grad = np.zeros_like(params)
    grad[0::3] = grad_x
    grad[1::3] = grad_y
    grad[2::3] = grad_r - 1.0  # -1 from -sum(r)
    
    return obj, grad


def construct():
    start_time = time.time()
    deadline = 120.0  # safety margin
    
    # Initial guess: grid layout
    side = int(np.ceil(np.sqrt(N)))
    init_params = []
    for i in range(side):
        for j in range(side):
            if len(init_params) >= 3 * N:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 0.001
            init_params.extend([x, y, r])
    init_params = np.array(init_params[:3*N])
    
    best_params = init_params.copy()
    best_obj, _ = _objective_and_grad(init_params)
    
    # Try multiple random restarts
    n_restarts = 5
    rng = np.random.RandomState(42)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
            
        if restart == 0:
            x0 = init_params.copy()
        else:
            # Random perturbation around grid
            x0 = init_params.copy()
            x0 += rng.uniform(-0.1, 0.1, size=x0.shape)
            # Ensure radii are positive
            x0[2::3] = np.maximum(0.01, x0[2::3])
        
        try:
            res = minimize(_objective_and_grad, x0, method='L-BFGS-B', 
                          jac=True, options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            if time.time() > deadline - 30:
                break
            
            # Check if result is better
            obj_val, _ = _objective_and_grad(res.x)
            # Only accept if penalty is small (valid solution)
            x = res.x[0::3]
            y = res.x[1::3]
            r = res.x[2::3]
            
            # Check validity
            valid = True
            for i in range(N):
                if r[i] <= 0 or x[i] < r[i] - 1e-6 or x[i] > 1 - r[i] + 1e-6 or \
                   y[i] < r[i] - 1e-6 or y[i] > 1 - r[i] + 1e-6:
                    valid = False
                    break
            if valid:
                for i in range(N):
                    for j in range(i+1, N):
                        dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                        if dist < r[i] + r[j] - 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid and obj_val < best_obj:
                best_obj = obj_val
                best_params = res.x.copy()
                
        except Exception:
            continue
    
    # Final polish with tighter tolerances
    if time.time() < deadline - 15:
        try:
            res = minimize(_objective_and_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
            obj_val, _ = _objective_and_grad(res.x)
            x = res.x[0::3]
            y = res.x[1::3]
            r = res.x[2::3]
            
            valid = True
            for i in range(N):
                if r[i] <= 0 or x[i] < r[i] - 1e-6 or x[i] > 1 - r[i] + 1e-6 or \
                   y[i] < r[i] - 1e-6 or y[i] > 1 - r[i] + 1e-6:
                    valid = False
                    break
            if valid:
                for i in range(N):
                    for j in range(i+1, N):
                        dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                        if dist < r[i] + r[j] - 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid and obj_val < best_obj:
                best_obj = obj_val
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Convert back to (x, y, r) array
    x = best_params[0::3]
    y = best_params[1::3]
    r = best_params[2::3]
    
    # Clamp to ensure validity
    r = np.maximum(1e-6, r)
    x = np.clip(x, r, 1 - r)
    y = np.clip(y, r, 1 - r)
    
    result = np.column_stack([x, y, r])
    return result

# EVOLVE-BLOCK-END
