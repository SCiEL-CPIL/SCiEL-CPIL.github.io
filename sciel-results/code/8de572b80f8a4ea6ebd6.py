"""Circle packing in perimeter-4 rectangle: force-directed + SLSQP optimization."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21
START = time.time()
DEADLINE = START + 135.0  # safety margin before 150s kill

def _time_ok():
    return time.time() < DEADLINE

def _rect_from_width(W):
    H = 2.0 - W
    return W, H

def _force_relax(x, W, H, n_iters=200, dt=0.01):
    """Force-directed relaxation to resolve overlaps."""
    cx = x[:, 0].copy()
    cy = x[:, 1].copy()
    r = x[:, 2].copy()
    n = len(cx)
    
    for it in range(n_iters):
        if not _time_ok():
            break
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        for i in range(n):
            # Wall forces
            if cx[i] < r[i]:
                fx[i] += (r[i] - cx[i]) * 10
            if cx[i] > W - r[i]:
                fx[i] -= (cx[i] - (W - r[i])) * 10
            if cy[i] < r[i]:
                fy[i] += (r[i] - cy[i]) * 10
            if cy[i] > H - r[i]:
                fy[i] -= (cy[i] - (H - r[i])) * 10
            
            # Circle-circle repulsion
            for j in range(n):
                if i == j:
                    continue
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d = np.hypot(dx, dy)
                if d < 1e-9:
                    dx, dy = 0.01, 0.01
                    d = np.hypot(dx, dy)
                overlap = r[i] + r[j] - d
                if overlap > 0:
                    fx[i] += (dx / d) * overlap * 10
                    fy[i] += (dy / d) * overlap * 10
        
        cx += dt * fx
        cy += dt * fy
        cx = np.clip(cx, 0, W)
        cy = np.clip(cy, 0, H)
    
    return np.column_stack([cx, cy, r])

def _slsqp_optimize(x, W_init, H_init, maxiter=100):
    """SLSQP optimization with free aspect ratio."""
    n = N
    
    def objective(params):
        W = params[0]
        H = params[1]
        cx = params[2:2+n]
        cy = params[2+n:2+2*n]
        r = params[2+2*n:2+3*n]
        
        # Perimeter constraint: 2*(W+H) = 4 => W+H = 2
        if abs(W + H - 2.0) > 0.01:
            return 1e10
        
        if W <= 0 or H <= 0:
            return 1e10
        
        # Wall constraints
        for i in range(n):
            if cx[i] < r[i] or cx[i] > W - r[i] or cy[i] < r[i] or cy[i] > H - r[i]:
                return 1e10
        
        # Circle-circle constraints
        for i in range(n):
            for j in range(i+1, n):
                d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
                if d < r[i] + r[j] - 1e-8:
                    return 1e10
        
        # Maximize sum of radii => minimize negative sum
        return -np.sum(r)
    
    def constraint_perimeter(params):
        W = params[0]
        H = params[1]
        return W + H - 2.0
    
    def constraint_walls(params, i, axis):
        W = params[0]
        H = params[1]
        cx = params[2:2+n]
        cy = params[2+n:2+2*n]
        r = params[2+2*n:2+3*n]
        if axis == 0:
            return cx[i] - r[i]
        elif axis == 1:
            return W - cx[i] - r[i]
        elif axis == 2:
            return cy[i] - r[i]
        else:
            return H - cy[i] - r[i]
    
    def constraint_circles(params, i, j):
        cx = params[2:2+n]
        cy = params[2+n:2+2*n]
        r = params[2+2*n:2+3*n]
        d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
        return d - r[i] - r[j]
    
    # Build constraints
    constraints = [{'type': 'eq', 'fun': constraint_perimeter}]
    
    # Wall constraints
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: constraint_walls(p, i, 0)})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: constraint_walls(p, i, 1)})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: constraint_walls(p, i, 2)})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: constraint_walls(p, i, 3)})
    
    # Circle-circle constraints
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i, j=j: constraint_circles(p, i, j)})
    
    # Initial params
    cx0 = x[:, 0]
    cy0 = x[:, 1]
    r0 = x[:, 2]
    x0 = np.concatenate([[W_init, H_init], cx0, cy0, r0])
    
    # Bounds
    bounds = [(0.1, 1.9), (0.1, 1.9)]
    bounds += [(0, W_init)] * n
    bounds += [(0, H_init)] * n
    bounds += [(0, 0.5)] * n
    
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                          constraints=constraints, maxiter=maxiter, ftol=1e-8)
        if result.success or result.fun < 1e9:
            W_opt = result.x[0]
            H_opt = result.x[1]
            cx_opt = result.x[2:2+n]
            cy_opt = result.x[2+n:2+2*n]
            r_opt = result.x[2+2*n:2+3*n]
            x_opt = np.column_stack([cx_opt, cy_opt, r_opt])
            return x_opt, W_opt, H_opt, -result.fun
        else:
            return x, W_init, H_init, 0.0
    except Exception:
        return x, W_init, H_init, 0.0

def _grid_initialization(W, H, seed):
    """Initialize circles in a grid pattern."""
    random.seed(seed)
    np.random.seed(seed)
    
    side = 5
    cx = np.zeros(N)
    cy = np.zeros(N)
    r = np.zeros(N)
    
    for i in range(N):
        row = i // side
        col = i % side
        cx[i] = (col + 0.5) / side * W + random.gauss(0, 0.02)
        cy[i] = (row + 0.5) / side * H + random.gauss(0, 0.02)
        r[i] = 0.5 / side * min(W, H) * 0.9
    
    return np.column_stack([cx, cy, r])

def _polish_result(x, W, H):
    """Polish radii by growing them as much as possible."""
    cx = x[:, 0].copy()
    cy = x[:, 1].copy()
    r = x[:, 2].copy()
    n = len(cx)
    
    # Multiple passes of radius growth
    for _ in range(5):
        if not _time_ok():
            break
        for i in range(n):
            max_r = min(cx[i], W - cx[i], cy[i], H - cy[i])
            for j in range(n):
                if i == j:
                    continue
                d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
                max_r = min(max_r, d - r[j])
            r[i] = max(0.01, max_r)
    
    return np.column_stack([cx, cy, r])

def _evaluate(x, W, H):
    """Return sum of radii if x is a valid packing, else None."""
    cx = x[:, 0].copy()
    cy = x[:, 1].copy()
    r = x[:, 2].copy()
    n = len(cx)
    cx = np.clip(cx, 0.0, W)
    cy = np.clip(cy, 0.0, H)
    r = np.maximum(r, 1e-6)
    lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
    r = np.minimum(r, lo)
    r = np.maximum(r, 1e-6)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
            if d + 1e-7 < r[i] + r[j]:
                return None
    return float(np.sum(r))

def construct():
    best_sum = 0.0
    best_arr = None
    best_W = 1.0
    best_H = 1.0
    
    random.seed(42)
    np.random.seed(42)
    
    # Try different aspect ratios
    Ws = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
    
    for W_init in Ws:
        if not _time_ok():
            break
        H_init = 2.0 - W_init
        
        for seed in [42, 123, 456]:
            if not _time_ok():
                break
            
            # 1. Initialize
            x_init = _grid_initialization(W_init, H_init, seed)
            
            # 2. Force Relaxation
            x_relaxed = _force_relax(x_init, W_init, H_init, n_iters=150)
            
            # 3. SLSQP Optimization
            x_opt, W_opt, H_opt, score = _slsqp_optimize(x_relaxed, W_init, H_init)
            
            # 4. Polish
            x_polished = _polish_result(x_opt, W_opt, H_opt)
            
            # 5. Validate and Update
            valid_score = _evaluate(x_polished, W_opt, H_opt)
            if valid_score is not None and valid_score > best_sum:
                best_sum = valid_score
                best_arr = x_polished
                best_W = W_opt
                best_H = H_opt
    
    # Fallback
    if best_arr is None:
        W, H = 1.0, 1.0
        x_init = _grid_initialization(W, H, 42)
        x_relaxed = _force_relax(x_init, W, H, n_iters=50)
        best_arr, best_W, best_H, _ = _slsqp_optimize(x_relaxed, W, H)
        if _evaluate(best_arr, best_W, best_H) is None:
             # Ultimate fallback: small circles in grid
             side = 5
             cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4)
                      for i in range(side) for j in range(side)]
             return np.array(cells[:N])
    
    # Final clamping to ensure validity
    cx = np.clip(best_arr[:, 0], 0.0, best_W)
    cy = np.clip(best_arr[:, 1], 0.0, best_H)
    r = best_arr[:, 2].copy()
    r = np.maximum(r, 1e-6)
    lo = np.minimum(np.minimum(cx, cy), np.minimum(best_W - cx, best_H - cy))
    r = np.minimum(r, lo)
    r = np.maximum(r, 1e-6)
    out = np.column_stack([cx, cy, r])
    
    return out
