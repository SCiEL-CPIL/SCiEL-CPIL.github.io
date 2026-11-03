"""Rewritten program for circle packing in perimeter-4 rectangle."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE = 145.0  # Safety margin before 150s kill

def _hex_init(W, H):
    """Generate a hexagonal packing initial guess."""
    # Estimate radius based on area
    # Area = W*H. Pack fraction ~0.9 for hex.
    # N * pi * r^2 ~ 0.9 * W * H
    r_est = np.sqrt(0.9 * W * H / (N * np.pi))
    
    # Hexagonal lattice spacing
    # Distance between centers = 2r
    # Row height = r * sqrt(3)
    rows = int(np.ceil(H / (r_est * np.sqrt(3))))
    cols = int(np.ceil(W / (2 * r_est)))
    
    circles = []
    r = r_est * 0.9 # Slightly smaller to ensure fit initially
    dy = r * np.sqrt(3)
    dx = 2 * r
    
    for i in range(rows):
        for j in range(cols):
            if len(circles) >= N:
                break
            x = (j + 0.5) * dx
            if i % 2 == 1:
                x += dx / 2
            y = (i + 0.5) * dy
            
            # Clamp to bounds
            x = np.clip(x, r, W - r)
            y = np.clip(y, r, H - r)
            
            # Ensure we don't go out of bounds in x
            if x + r > W or x - r < 0:
                continue
            if y + r > H or y - r < 0:
                continue
                
            circles.append([x, y, r])
            
    # If we didn't get N circles, fill the rest with small circles in a grid
    if len(circles) < N:
        # Simple grid fallback for remaining
        side = int(np.ceil(np.sqrt(N)))
        for i in range(side):
            for j in range(side):
                if len(circles) >= N:
                    break
                cx = (j + 0.5) * W / side
                cy = (i + 0.5) * H / side
                r_small = min(W, H) / (2 * side) * 0.9
                cx = np.clip(cx, r_small, W - r_small)
                cy = np.clip(cy, r_small, H - r_small)
                circles.append([cx, cy, r_small])
                
    return np.array(circles[:N])

def _objective_and_grad(params, W, H):
    """
    Objective: -sum(r)
    Gradients: [0, 0, -1] for each circle
    """
    n = N
    circles = params.reshape(n, 3)
    r = circles[:, 2]
    obj = -np.sum(r)
    grad = np.zeros_like(params)
    grad[2::3] = -1.0
    return obj, grad

def _constraints_and_grads(params, W, H):
    """
    Returns list of constraint functions and their gradients.
    Each constraint is g(params) >= 0.
    """
    n = N
    circles = params.reshape(n, 3)
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    cons = []
    
    # 1. Boundary Constraints
    # x - r >= 0
    cons.append({
        'type': 'ineq',
        'fun': lambda p, x=x, r=r: x - r, # Note: x, r from closure, but SLSQP passes p. 
        # Wait, lambda captures x, r from outer scope which is stale if p changes.
        # We must define fun to calculate from p.
        'jac': lambda p: np.zeros_like(p) # Placeholder, will fix
    })
    
    # This approach of creating lambdas in a loop is inefficient and error-prone with closures.
    # Better to define a single function that returns all constraints and their Jacobian.
    # However, scipy minimize expects a list of dicts or a single dict with 'fun' returning array.
    # Let's use the array format for constraints.
    
    # We will construct the arrays directly in the wrapper functions passed to minimize.
    
    return None # Placeholder

def _cons_fun(params, W, H):
    """
    Returns array of constraint values g_i >= 0.
    """
    n = N
    circles = params.reshape(n, 3)
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    vals = []
    
    # Boundaries
    vals.extend(x - r)
    vals.extend(W - x - r)
    vals.extend(y - r)
    vals.extend(H - y - r)
    vals.extend(r - 1e-6) # r > 0
    
    # Overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx*dx + dy*dy
            min_dist = r[i] + r[j]
            # Constraint: dist_sq - min_dist^2 >= 0
            vals.append(dist_sq - min_dist*min_dist)
            
    return np.array(vals)

def _cons_jac(params, W, H):
    """
    Returns Jacobian matrix of constraints.
    Shape: (num_constraints, num_params)
    """
    n = N
    circles = params.reshape(n, 3)
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    num_cons = 5*n + n*(n-1)//2
    jac = np.zeros((num_cons, 3*n))
    
    idx = 0
    # 1. x - r >= 0
    for i in range(n):
        jac[idx, 3*i] = 1.0
        jac[idx, 3*i+2] = -1.0
        idx += 1
    # 2. W - x - r >= 0
    for i in range(n):
        jac[idx, 3*i] = -1.0
        jac[idx, 3*i+2] = -1.0
        idx += 1
    # 3. y - r >= 0
    for i in range(n):
        jac[idx, 3*i+1] = 1.0
        jac[idx, 3*i+2] = -1.0
        idx += 1
    # 4. H - y - r >= 0
    for i in range(n):
        jac[idx, 3*i+1] = -1.0
        jac[idx, 3*i+2] = -1.0
        idx += 1
    # 5. r - 1e-6 >= 0
    for i in range(n):
        jac[idx, 3*i+2] = 1.0
        idx += 1
        
    # 6. Overlaps: (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            ri = r[i]
            rj = r[j]
            
            # d/dxi: 2*dx
            # d/dyi: 2*dy
            # d/dri: -2*ri
            # d/dxj: -2*dx
            # d/dyj: -2*dy
            # d/drj: -2*rj
            
            jac[idx, 3*i] = 2*dx
            jac[idx, 3*i+1] = 2*dy
            jac[idx, 3*i+2] = -2*ri
            jac[idx, 3*j] = -2*dx
            jac[idx, 3*j+1] = -2*dy
            jac[idx, 3*j+2] = -2*rj
            idx += 1
            
    return jac

def _penalty_objective(params, W, H, weight=100.0):
    """
    Objective for L-BFGS-B: -sum(r) + weight * sum(soft_violations)
    """
    n = N
    circles = params.reshape(n, 3)
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    obj = -np.sum(r)
    penalty = 0.0
    
    # Boundary penalties
    # x - r < 0 -> (r-x)^2
    penalty += np.sum(np.maximum(0, r - x)**2)
    penalty += np.sum(np.maximum(0, x + r - W)**2)
    penalty += np.sum(np.maximum(0, r - y)**2)
    penalty += np.sum(np.maximum(0, y + r - H)**2)
    
    # Overlap penalties
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                penalty += (min_dist - dist)**2
                
    return obj + weight * penalty

def _penalty_grad(params, W, H, weight=100.0):
    """
    Gradient of penalty objective.
    """
    n = N
    circles = params.reshape(n, 3)
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    grad = np.zeros_like(params)
    grad[2::3] = -1.0 # d(-sum r)/dr
    
    # Boundary gradients
    # d/dx (max(0, r-x)^2) = -2*max(0, r-x)
    v = np.maximum(0, r - x)
    grad[0::3] += weight * (-2 * v)
    grad[2::3] += weight * (2 * v)
    
    v = np.maximum(0, x + r - W)
    grad[0::3] += weight * (2 * v)
    grad[2::3] += weight * (2 * v)
    
    v = np.maximum(0, r - y)
    grad[1::3] += weight * (-2 * v)
    grad[2::3] += weight * (2 * v)
    
    v = np.maximum(0, y + r - H)
    grad[1::3] += weight * (2 * v)
    grad[2::3] += weight * (2 * v)
    
    # Overlap gradients
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx*dx + dy*dy
            if dist_sq < 1e-12:
                continue
            dist = np.sqrt(dist_sq)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                diff = min_dist - dist
                # Penalty = diff^2
                # d/dx_i = 2*diff * d(diff)/dx_i
                # d(diff)/dx_i = -dx/dist
                factor = 2 * weight * diff
                grad[3*i] += factor * (-dx / dist)
                grad[3*i+1] += factor * (-dy / dist)
                grad[3*i+2] += factor * 1.0
                grad[3*j] += factor * (dx / dist)
                grad[3*j+1] += factor * (dy / dist)
                grad[3*j+2] += factor * 1.0
                
    return grad

def _check_validity(circles, W, H):
    if len(circles) != N:
        return False
    for i in range(N):
        x, y, r = circles[i]
        if r <= 1e-6:
            return False
        if x - r < -1e-5 or y - r < -1e-5 or x + r > W + 1e-5 or y + r > H + 1e-5:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i][2] + circles[j][2] - 1e-5:
                return False
    return True

def _solve_aspect(W, H, start_time, max_time=10.0):
    """
    Solve for a specific aspect ratio.
    Returns (best_circles, best_score).
    """
    start = time.time()
    if time.time() - start_time > DEADLINE - 5:
        return None, -np.inf
        
    init = _hex_init(W, H)
    best_circles = init.copy()
    best_score = np.sum(init[:, 2])
    
    # Phase 1: Fast L-BFGS-B with penalty
    params = init.flatten()
    bounds = [(1e-6, W-1e-6), (1e-6, H-1e-6), (1e-6, min(W,H)/2)] * N
    
    try:
        res_lbfgs = minimize(
            _penalty_objective, params, 
            jac=_penalty_grad,
            args=(W, H, 100.0),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-8, 'maxls': 50}
        )
        circles_lbfgs = res_lbfgs.x.reshape(N, 3)
        if _check_validity(circles_lbfgs, W, H):
            s = np.sum(circles_lbfgs[:, 2])
            if s > best_score:
                best_score = s
                best_circles = circles_lbfgs.copy()
    except Exception:
        pass
    
    # Phase 2: Precise SLSQP with constraints
    if time.time() - start_time < DEADLINE - 10:
        try:
            # Start SLSQP from the best of init or lbfgs
            start_params = best_circles.flatten()
            
            cons = {
                'type': 'ineq',
                'fun': lambda p: _cons_fun(p, W, H),
                'jac': lambda p: _cons_jac(p, W, H)
            }
            
            res_slsqp = minimize(
                _objective_and_grad, start_params,
                args=(W, H),
                method='SLSQP',
                jac=True,
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            circles_slsqp = res_slsqp.x.reshape(N, 3)
            if _check_validity(circles_slsqp, W, H):
                s = np.sum(circles_slsqp[:, 2])
                if s > best_score:
                    best_score = s
                    best_circles = circles_slsqp.copy()
        except Exception:
            pass
            
    return best_circles, best_score

def construct():
    start_time = time.time()
    best_result = None
    best_score = -np.inf
    best_W, best_H = 1.0, 1.0
    
    # Search aspect ratios
    # W + H = 2
    # Try W from 0.8 to 1.2
    aspect_ratios = [0.8, 0.9, 1.0, 1.1, 1.2]
    
    for W in aspect_ratios:
        H = 2.0 - W
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 15:
            break
            
        circles, score = _solve_aspect(W, H, start_time)
        if circles is not None and score > best_score:
            best_score = score
            best_result = circles
            best_W, best_H = W, H
            
    # If no result, use a safe grid
    if best_result is None:
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cx = (j + 0.5) / side
                cy = (i + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append([cx, cy, r])
        best_result = np.array(cells[:N])
        best_W, best_H = 1.0, 1.0
        best_score = np.sum(best_result[:, 2])
        
    return best_result

