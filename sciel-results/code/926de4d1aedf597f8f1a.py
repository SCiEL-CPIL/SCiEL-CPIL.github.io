import numpy as np
import time
import math
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START

def _soft_min(a, b, beta=100.0):
    """Smooth approximation of min(a, b) using log-sum-exp."""
    m = np.minimum(a, b)
    return m - np.logaddexp(-beta*(a-m), -beta*(b-m)) / beta

def _objective_and_grad(vars, W, lam_overlap, lam_bound, beta=50.0):
    """
    vars: [x1, y1, r1, ..., xN, yN, rN]
    W: width
    Returns (objective, gradient) with soft-min overlap penalty.
    """
    H = 2.0 - W
    x = vars[0:N]
    y = vars[N:2*N]
    r = vars[2*N:3*N]
    
    grad = np.zeros(3*N)
    
    # Base objective: maximize sum of radii -> minimize -sum(r)
    val = -np.sum(r)
    grad[2*N:3*N] -= 1.0
    
    # Overlap penalty using soft-min
    # For each pair (i,j), penalty = max(0, (ri+rj) - dist_ij)^2
    # We use a smooth version: let g_ij = (ri+rj) - dist_ij
    # penalty = softplus(g_ij * beta) / beta * g_ij (smooth max)
    # Simpler: use (max(0, g_ij))^2 with smoothed max via log-sum-exp
    
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d_sq = dx*dx + dy*dy
            d = math.sqrt(d_sq + 1e-12)
            min_d = r[i] + r[j]
            g = min_d - d  # positive when overlapping
            
            if g > 0:
                # Smooth penalty: g^2 with gradient
                pen = g * g
                val += lam_overlap * pen
                
                # Gradients
                # d(pen)/d(dx) = 2*g*(-1) * d(d)/d(dx) = 2*g*(dx/d)
                # Actually: g = min_d - d, so dg/dx_i = -dx/d, dg/dx_j = dx/d
                # d(pen)/dx_i = 2*g * (-dx/d) = -2*g*dx/d
                # d(pen)/dx_j = 2*g * (dx/d) = 2*g*dx/d
                # Similarly for y
                # d(pen)/dr_i = 2*g * 1 = 2*g
                # d(pen)/dr_j = 2*g * 1 = 2*g
                
                coeff = 2.0 * g / d
                gx_i = -coeff * dx
                gx_j = coeff * dx
                gy_i = -coeff * dy
                gy_j = coeff * dy
                gr = 2.0 * g
                
                grad[i] += lam_overlap * gx_i
                grad[j] += lam_overlap * gx_j
                grad[N+i] += lam_overlap * gy_i
                grad[N+j] += lam_overlap * gy_j
                grad[2*N+i] += lam_overlap * gr
                grad[2*N+j] += lam_overlap * gr
    
    # Boundary penalty
    for i in range(N):
        # x violation
        if x[i] < r[i]:
            v = r[i] - x[i]
            val += lam_bound * v * v
            grad[i] += 2.0 * lam_bound * v
            grad[2*N+i] -= 2.0 * lam_bound * v
        elif x[i] > W - r[i]:
            v = x[i] - (W - r[i])
            val += lam_bound * v * v
            grad[i] -= 2.0 * lam_bound * v
            grad[2*N+i] += 2.0 * lam_bound * v
            
        # y violation
        if y[i] < r[i]:
            v = r[i] - y[i]
            val += lam_bound * v * v
            grad[N+i] += 2.0 * lam_bound * v
            grad[2*N+i] -= 2.0 * lam_bound * v
        elif y[i] > H - r[i]:
            v = y[i] - (H - r[i])
            val += lam_bound * v * v
            grad[N+i] -= 2.0 * lam_bound * v
            grad[2*N+i] += 2.0 * lam_bound * v
            
        # radius lower bound
        if r[i] < 1e-5:
            v = 1e-5 - r[i]
            val += lam_bound * v * v
            grad[2*N+i] -= 2.0 * lam_bound * v

    return val, grad

def _compute_value(x, y, r, W):
    """
    Returns sum of radii if valid, else 0.0
    """
    H = 2.0 - W
    if W <= 0 or H <= 0:
        return 0.0
        
    # Check bounds
    for i in range(N):
        if x[i] < r[i] - 1e-7 or x[i] > W - r[i] + 1e-7:
            return 0.0
        if y[i] < r[i] - 1e-7 or y[i] > H - r[i] + 1e-7:
            return 0.0
            
    # Check pairwise distances
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx*dx + dy*dy
            min_dist = r[i] + r[j]
            if dist_sq < min_dist*min_dist - 1e-7:
                return 0.0
                
    return np.sum(r)

def _grid_init(W, H, n_cols, n_rows):
    """
    Initialize circles in a grid.
    """
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    
    idx = 0
    for row in range(n_rows):
        for col in range(n_cols):
            if idx >= N:
                break
            cx = (col + 0.5) * (W / n_cols)
            cy = (row + 0.5) * (H / n_rows)
            
            # Conservative radius
            dx_cell = W / n_cols
            dy_cell = H / n_rows
            r_max = 0.5 * min(dx_cell, dy_cell) * 0.95
            
            x[idx] = cx
            y[idx] = cy
            r[idx] = r_max
            idx += 1
        if idx >= N:
            break
            
    return x, y, r

def _run_optimization(x0, y0, r0, W0, deadline, lam_overlap=100.0, lam_bound=100.0, maxiter=500):
    """
    Run L-BFGS-B optimization with penalty and analytical gradient.
    """
    H0 = 2.0 - W0
    vars0 = np.concatenate([x0, y0, r0])
    
    # Bounds for L-BFGS-B
    bounds = []
    for i in range(N):
        bounds.append((1e-5, W0 - 1e-5)) # x
    for i in range(N):
        bounds.append((1e-5, H0 - 1e-5)) # y
    for i in range(N):
        bounds.append((1e-5, 0.5)) # r
        
    options = {
        'maxiter': maxiter,
        'ftol': 1e-12,
        'gtol': 1e-10,
        'disp': False
    }
    
    start_time = time.time()
    
    try:
        def obj_and_grad(v):
            if time.time() > deadline:
                raise TimeoutError()
            return _objective_and_grad(v, W0, lam_overlap, lam_bound)
        
        res = minimize(
            lambda v: obj_and_grad(v)[0],
            vars0,
            jac=lambda v: obj_and_grad(v)[1],
            method='L-BFGS-B',
            bounds=bounds,
            options=options
        )
        
        vars_opt = res.x
        x_opt = vars_opt[0:N]
        y_opt = vars_opt[N:2*N]
        r_opt = vars_opt[2*N:3*N]
        
        # Validate
        val = _compute_value(x_opt, y_opt, r_opt, W0)
        
        if val > 0:
            return x_opt, y_opt, r_opt, W0, val
        else:
            return x0, y0, r0, W0, _compute_value(x0, y0, r0, W0)
            
    except Exception as e:
        return x0, y0, r0, W0, _compute_value(x0, y0, r0, W0)

def _polish(x, y, r, W, deadline):
    """
    Final high-precision polish pass with tighter tolerances.
    """
    H = 2.0 - W
    vars0 = np.concatenate([x, y, r])
    
    bounds = []
    for i in range(N):
        bounds.append((max(1e-5, x[i] - 0.005), min(W - 1e-5, x[i] + 0.005)))
    for i in range(N):
        bounds.append((max(1e-5, y[i] - 0.005), min(H - 1e-5, y[i] + 0.005)))
    for i in range(N):
        bounds.append((max(1e-5, r[i] - 0.005), min(0.5, r[i] + 0.005)))
        
    lam_overlap = 500.0
    lam_bound = 500.0
    
    options = {
        'maxiter': 1000,
        'ftol': 1e-14,
        'gtol': 1e-12,
        'disp': False
    }
    
    start_time = time.time()
    
    try:
        def obj_and_grad(v):
            if time.time() > deadline:
                raise TimeoutError()
            return _objective_and_grad(v, W, lam_overlap, lam_bound)
        
        res = minimize(
            lambda v: obj_and_grad(v)[0],
            vars0,
            jac=lambda v: obj_and_grad(v)[1],
            method='L-BFGS-B',
            bounds=bounds,
            options=options
        )
        
        vars_opt = res.x
        x_opt = vars_opt[0:N]
        y_opt = vars_opt[N:2*N]
        r_opt = vars_opt[2*N:3*N]
        
        val = _compute_value(x_opt, y_opt, r_opt, W)
        
        if val > 0:
            return x_opt, y_opt, r_opt, val
        else:
            return x, y, r, _compute_value(x, y, r, W)
            
    except Exception as e:
        return x, y, r, _compute_value(x, y, r, W)

def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    
    best_val = 0.0
    best_x = np.zeros(N)
    best_y = np.zeros(N)
    best_r = np.zeros(N)
    best_W = 1.0
    
    # Try different aspect ratios
    aspect_ratios = [0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]
    
    # Grid configurations (cols, rows)
    grid_configs = [(4, 6), (5, 5), (6, 4), (3, 7), (7, 3)]
    
    # First pass: find a good aspect ratio
    for W0 in aspect_ratios:
        if time.time() > deadline - 25:
            break
        H0 = 2.0 - W0
        
        for n_cols, n_rows in grid_configs:
            if time.time() > deadline - 25:
                break
                
            x0, y0, r0 = _grid_init(W0, H0, n_cols, n_rows)
            
            # Add small noise
            x0 += np.random.uniform(-0.01, 0.01, N)
            y0 += np.random.uniform(-0.01, 0.01, N)
            r0 += np.random.uniform(-0.01, 0.01, N)
            
            # Clamp
            for i in range(N):
                r0[i] = max(1e-4, r0[i])
                x0[i] = np.clip(x0[i], r0[i], W0 - r0[i])
                y0[i] = np.clip(y0[i], r0[i], H0 - r0[i])
                
            x_opt, y_opt, r_opt, W_opt, val = _run_optimization(x0, y0, r0, W0, deadline)
            
            if val > best_val:
                best_val = val
                best_x = x_opt.copy()
                best_y = y_opt.copy()
                best_r = r_opt.copy()
                best_W = W_opt
    
    # Second pass: refine the best aspect ratio with more restarts
    if time.time() < deadline - 20 and best_val > 0:
        W_best = best_W
        H_best = 2.0 - W_best
        
        n_restarts = 10
        for _ in range(n_restarts):
            if time.time() > deadline - 20:
                break
                
            n_cols, n_rows = (5, 5) if abs(W_best - 1.0) < 0.1 else (4, 6) if W_best < 1.0 else (6, 4)
            x0, y0, r0 = _grid_init(W_best, H_best, n_cols, n_rows)
            
            x0 += np.random.uniform(-0.03, 0.03, N)
            y0 += np.random.uniform(-0.03, 0.03, N)
            r0 += np.random.uniform(-0.02, 0.02, N)
            
            for i in range(N):
                r0[i] = max(1e-4, r0[i])
                x0[i] = np.clip(x0[i], r0[i], W_best - r0[i])
                y0[i] = np.clip(y0[i], r0[i], H_best - r0[i])
                
            x_opt, y_opt, r_opt, W_opt, val = _run_optimization(x0, y0, r0, W_best, deadline)
            
            if val > best_val:
                best_val = val
                best_x = x_opt.copy()
                best_y = y_opt.copy()
                best_r = r_opt.copy()
                best_W = W_opt
    
    # Final polish pass
    if time.time() < deadline - 5 and best_val > 0:
        x_pol, y_pol, r_pol, val_pol = _polish(best_x, best_y, best_r, best_W, deadline)
        if val_pol > best_val:
            best_x = x_pol
            best_y = y_pol
            best_r = r_pol
            best_val = val_pol
    
    # Fallback
    if best_val == 0.0:
        W = 1.0
        H = 1.0
        side = 5
        for i in range(N):
            row = i // 5
            col = i % 5
            best_x[i] = (col + 0.5) / side
            best_y[i] = (row + 0.5) / side
            best_r[i] = 0.5 / side - 1e-4
        best_W = W
        
    return np.column_stack([best_x, best_y, best_r])

# EVOLVE-BLOCK-END
