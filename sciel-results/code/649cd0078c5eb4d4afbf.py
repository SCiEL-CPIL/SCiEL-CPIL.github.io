"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random
from scipy.optimize import minimize

def _soft_objective_and_grad(params, n=N):
    """
    Objective: minimize -sum(r) + penalty for violations.
    params: [x0, y0, r0, x1, y1, r1, ...]
    """
    xy = params[0::3].reshape(-1, 2)
    r = params[1::3].reshape(-1)
    
    # Ensure radii are positive for calculation, though optimizer handles bounds
    r = np.maximum(r, 1e-6)
    
    obj = -np.sum(r)
    grad = np.zeros_like(params)
    
    # Gradient of objective: -1 for all radii
    grad[1::3] = -1.0
    
    # Wall penalties
    # x - r < 0 => r - x > 0
    # (1-x) - r < 0 => x + r - 1 > 0
    # y - r < 0 => r - y > 0
    # (1-y) - r < 0 => y + r - 1 > 0
    
    # Left wall: violation = max(0, r - x)
    # d/dx: -1 if active, d/dr: 1 if active
    # Right wall: violation = max(0, x + r - 1)
    # d/dx: 1 if active, d/dr: 1 if active
    
    # Bottom: max(0, r - y)
    # Top: max(0, y + r - 1)
    
    # We use a soft penalty: sum of squares of violations? 
    # Or just linear penalty? Linear is non-differentiable at 0. 
    # Let's use a smooth approximation or just simple penalty with subgradient?
    # For L-BFGS-B, we need differentiability. 
    # Let's use penalty = sum(violation^2) * weight
    # Actually, for packing, a large linear penalty often works if we handle the kink carefully,
    # but let's stick to a differentiable soft constraint:
    # f(v) = 0.5 * max(0, v)^2 is differentiable (grad is max(0,v)).
    
    penalty_weight = 100.0
    
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        
        # Left
        v = ri - x
        if v > 0:
            obj += penalty_weight * 0.5 * v * v
            grad[3*i + 1] += penalty_weight * v   # dr
            grad[3*i + 0] -= penalty_weight * v   # dx
            
        # Right
        v = x + ri - 1.0
        if v > 0:
            obj += penalty_weight * 0.5 * v * v
            grad[3*i + 1] += penalty_weight * v   # dr
            grad[3*i + 0] += penalty_weight * v   # dx
            
        # Bottom
        v = ri - y
        if v > 0:
            obj += penalty_weight * 0.5 * v * v
            grad[3*i + 1] += penalty_weight * v   # dr
            grad[3*i + 2] -= penalty_weight * v   # dy
            
        # Top
        v = y + ri - 1.0
        if v > 0:
            obj += penalty_weight * 0.5 * v * v
            grad[3*i + 1] += penalty_weight * v   # dr
            grad[3*i + 2] += penalty_weight * v   # dy

    # Pairwise penalties
    # dist < ri + rj => (ri + rj - dist)^2
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist < 1e-9:
                # Avoid division by zero, push apart randomly
                dist = 1e-9
                dx = 1e-9
                dy = 0.0
                
            v = r[i] + r[j] - dist
            if v > 0:
                obj += penalty_weight * 0.5 * v * v
                
                # Gradients
                # d/dri: 1
                # d/drj: 1
                # d/dxi: -dx/dist
                # d/dxj: dx/dist
                # d/dyi: -dy/dist
                # d/dyj: dy/dist
                
                grad[3*i + 1] += penalty_weight * v
                grad[3*j + 1] += penalty_weight * v
                
                gx = -dx / dist
                gy = -dy / dist
                
                grad[3*i + 0] += penalty_weight * v * gx
                grad[3*i + 2] += penalty_weight * v * gy
                grad[3*j + 0] += penalty_weight * v * (-gx)
                grad[3*j + 2] += penalty_weight * v * (-gy)
                
    return obj, grad

def _is_valid(xy, r, tol=1e-7):
    n = len(xy)
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if x - ri < -tol or 1.0 - x - ri < -tol or y - ri < -tol or 1.0 - y - ri < -tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True

def _largest_empty_circle(xy, r, n_samples=50):
    """
    Find the largest empty circle in the current packing to potentially add a new circle
    or reposition a small one. Here we use it for initialization diversity.
    Actually, for N=26, we just need good starts.
    """
    pass

def _init_grid():
    # 5x5 grid is 25 circles. Add 1 in center?
    # Or 5x5 with one removed and two added?
    # Let's try a 5x5 grid, remove corner, add 2 in center gaps?
    # Simpler: 5x5 grid, radii 0.1.
    xy = np.zeros((N, 2))
    r = np.zeros(N)
    
    # 5x5 grid
    side = 5
    for i in range(side):
        for j in range(side):
            idx = i * side + j
            if idx < N:
                xy[idx, 0] = (i + 0.5) / side
                xy[idx, 1] = (j + 0.5) / side
                r[idx] = 0.5 / side - 0.01 # 0.09
    
    # We have 25. Need 1 more.
    # Place it in the center? (0.5, 0.5) is occupied by idx 12.
    # Shift the center circle?
    # Or just use a 5x5 grid and shrink everything slightly to fit a 26th?
    # Better: Use a hexagonal-like packing for initialization.
    
    return xy, r

def _init_hexagonal():
    # Hexagonal packing of 26 circles
    # Rows: 5, 5, 5, 5, 4, 2? No, let's do 5,5,5,5,4,2? 
    # 5+5+5+5+4+2 = 26.
    # Or 5,5,5,5,5,1?
    # Let's try 5,5,5,5,4,2.
    xy = np.zeros((N, 2))
    r = np.zeros(N)
    
    rows = [5, 5, 5, 5, 4, 2]
    idx = 0
    y_spacing = 0.17
    x_spacing = 0.18
    offset = 0
    
    for i, count in enumerate(rows):
        y = 0.1 + i * y_spacing
        # Center the row
        start_x = (1.0 - (count - 1) * x_spacing) / 2.0
        if i % 2 == 1:
            start_x += x_spacing / 2.0
            
        for j in range(count):
            if idx >= N: break
            x = start_x + j * x_spacing
            xy[idx, 0] = x
            xy[idx, 1] = y
            r[idx] = 0.08 # Base radius
            idx += 1
            
    return xy, r

def _init_random():
    xy = np.random.rand(N, 2)
    r = np.full(N, 0.05)
    return xy, r

def _gradient_optimize(xy, r, deadline):
    """
    Run L-BFGS-B to maximize sum of radii subject to constraints.
    """
    n = N
    params = np.zeros(3 * n)
    params[0::3] = xy[:, 0]
    params[1::3] = xy[:, 1]
    params[2::3] = r
    
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.001, 0.4)) # r
    
    start_time = time.time()
    
    # We need to stop early if deadline is approached
    # L-BFGS-B doesn't have a simple time-based callback that aborts cleanly in all versions,
    # but we can check elapsed time in the objective? No, that's expensive.
    # We'll just run it and hope it's fast, or wrap it in a timeout if possible.
    # Since we can't easily timeout L-BFGS-B, we will limit iterations.
    
    res = minimize(
        _soft_objective_and_grad,
        params,
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
    )
    
    if time.time() > deadline:
        return xy, r, 0.0, False
        
    opt_params = res.x
    opt_xy = opt_params[0::3].reshape(-1, 2)
    opt_r = opt_params[1::3].reshape(-1)
    
    # Post-processing: project to feasible region if slightly invalid due to numerical errors
    # If invalid, shrink radii slightly
    valid = _is_valid(opt_xy, opt_r)
    if not valid:
        # Try to fix by shrinking
        for _ in range(10):
            opt_r *= 0.95
            valid = _is_valid(opt_xy, opt_r)
            if valid:
                break
        if not valid:
            return xy, r, 0.0, False
            
    total = np.sum(opt_r)
    return opt_xy, opt_r, total, valid

def _local_search(xy, r, deadline):
    """
    Simple local search to polish the gradient result.
    Try to increase radii slightly.
    """
    n = N
    current_xy = xy.copy()
    current_r = r.copy()
    current_total = np.sum(current_r)
    best_total = current_total
    best_xy = current_xy.copy()
    best_r = current_r.copy()
    
    step = 0.005
    for i in range(n):
        if time.time() > deadline:
            break
            
        # Try to increase radius of circle i
        old_r = current_r[i]
        new_r = old_r + step
        current_r[i] = new_r
        
        if _is_valid(current_xy, current_r):
            current_total += step
            best_total = current_total
            best_r = current_r.copy()
            best_xy = current_xy.copy()
        else:
            current_r[i] = old_r
            # If we can't increase, try moving it to make space?
            # For now, just skip.
            
    return best_xy, best_r, best_total, True

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    best_total = 0.0
    best_xy = None
    best_r = None
    
    inits = [
        _init_grid(),
        _init_hexagonal(),
        _init_random()
    ]
    
    for xy, r in inits:
        if time.time() > deadline - 20.0:
            break
            
        # Run gradient optimization
        opt_xy, opt_r, opt_total, valid = _gradient_optimize(xy, r, deadline - 5.0)
        
        if valid and opt_total > best_total:
            best_total = opt_total
            best_xy = opt_xy
            best_r = opt_r
            
        # Local search polish
        if valid and time.time() < deadline - 2.0:
            ls_xy, ls_r, ls_total, ls_valid = _local_search(opt_xy, opt_r, deadline - 1.0)
            if ls_valid and ls_total > best_total:
                best_total = ls_total
                best_xy = ls_xy
                best_r = ls_r
                
    # Fallback if nothing worked
    if best_xy is None:
        xy, r = _init_grid()
        # Shrink to ensure validity
        for _ in range(20):
            if _is_valid(xy, r):
                break
            r *= 0.9
        best_xy = xy
        best_r = r
        best_total = np.sum(r)
        
    # Final validation and clipping
    n = len(best_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)
        
    final_xy = result[:, :2]
    final_r = result[:, 2]
    
    if not _is_valid(final_xy, final_r):
        # Emergency fallback: tiny circles
        result = np.zeros((N, 3))
        for i in range(N):
            result[i, 0] = (i % 5 + 0.5) / 5.0
            result[i, 1] = (i // 5 + 0.5) / 5.0
            result[i, 2] = 0.05
        return result
        
    return result
# EVOLVE-BLOCK-END
