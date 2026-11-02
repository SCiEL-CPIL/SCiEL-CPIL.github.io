"""
Strategy: Simulated Annealing + L-BFGS-B Polish for Circle Packing
1. Jointly optimize circle positions, radii, and rectangle aspect ratio using SA.
2. SA uses a penalty-based energy function to handle constraints.
3. Final polish uses L-BFGS-B with softplus penalties to refine the best solution.
4. Ensures validity by projecting circles back into bounds if necessary.
"""

import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21

def _softplus(x, beta=100.0):
    """Numerically stable softplus."""
    # log(1 + exp(beta*x)) / beta
    # For large positive x, approx x
    # For large negative x, approx 0
    # Avoid overflow in exp
    x = np.asarray(x)
    out = np.zeros_like(x)
    mask = x > 0
    out[mask] = x[mask] / beta
    out[~mask] = np.log1p(np.exp(beta * x[~mask])) / beta
    return out

def _energy(x, aspect, beta=50.0):
    """
    Calculate energy (penalty - sum_radii). We minimize this.
    x: [x1, y1, r1, ..., xN, yN, rN]
    aspect: w/h
    """
    w = 2.0 * aspect / (1.0 + aspect)
    h = 2.0 / (1.0 + aspect)
    
    xy = x[0::3]
    yv = x[1::3]
    r = x[2::3]
    
    # 1. Boundary Penalties
    # Violations: x < r, x > w-r, y < r, y > h-r
    v_x_min = r - xy
    v_x_max = xy - (w - r)
    v_y_min = r - yv
    v_y_max = yv - (h - r)
    
    pen_b = (np.sum(_softplus(v_x_min, beta)) + np.sum(_softplus(v_x_max, beta)) +
             np.sum(_softplus(v_y_min, beta)) + np.sum(_softplus(v_y_max, beta)))
    
    # 2. Collision Penalties
    # Pre-computed pairs in closure
    i_idx, j_idx = _pairs
    dx = xy[i_idx] - xy[j_idx]
    dy = yv[i_idx] - yv[j_idx]
    dist = np.sqrt(dx*dx + dy*dy + 1e-12)
    
    r_sum = r[i_idx] + r[j_idx]
    v_coll = r_sum - dist
    
    pen_c = np.sum(_softplus(v_coll, beta))
    
    total_penalty = pen_b + pen_c
    sum_r = np.sum(r)
    
    return total_penalty - sum_r

def _check_validity(x, aspect, tol=1e-5):
    w = 2.0 * aspect / (1.0 + aspect)
    h = 2.0 / (1.0 + aspect)
    
    xy = x[0::3]
    yv = x[1::3]
    r = x[2::3]
    
    if np.any(r < 1e-6):
        return False
    if np.any(xy < r - tol) or np.any(xy > w - r + tol):
        return False
    if np.any(yv < r - tol) or np.any(yv > h - r + tol):
        return False
        # Collision check
    i_idx, j_idx = _pairs
    dx = xy[i_idx] - xy[j_idx]
    dy = yv[i_idx] - yv[j_idx]
    dist = np.sqrt(dx*dx + dy*dy)
    r_sum = r[i_idx] + r[j_idx]
    if np.any(dist < r_sum - tol):
        return False
        
    return True

def _get_score(x, aspect):
    if not _check_validity(x, aspect):
        return -np.inf
    return np.sum(x[2::3])

def _polish(x0, aspect0, time_limit=10.0):
    """
    Use L-BFGS-B to polish the solution.
    """
    start_time = time.time()
    
    def obj(x):
        return _energy(x, aspect0, beta=100.0)
    
    # We don't easily bound x in L-BFGS-B without knowing w,h exactly, 
    # but we can use the current w,h as bounds for x,y and [0, min(w,h)/2] for r.
    w = 2.0 * aspect0 / (1.0 + aspect0)
    h = 2.0 / (1.0 + aspect0)
    
    bounds = []
    for i in range(N):
        bounds.append((0, w)) # x
        bounds.append((0, h)) # y
        bounds.append((0, min(w, h) * 0.5)) # r
        
    try:
        res = minimize(obj, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 10000})
        if time.time() - start_time > time_limit:
            return x0
        return res.x
    except Exception:
        return x0

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    global _pairs
    i_idx, j_idx = np.triu_indices(N, k=1)
    _pairs = (i_idx, j_idx)
    
    best_score = -np.inf
    best_x = None
    best_aspect = None
    
    # SA Parameters
    T0 = 0.1
    T_end = 0.001
    alpha = 0.9995
    max_iter = 20000
    num_restarts = 10
    
    np.random.seed(42)
    random.seed(42)
    
    for restart in range(num_restarts):
        if time.time() > deadline - 20.0: # Leave time for polish
            break
            
        # Initialize
        # Try different aspect ratios
        aspect = np.random.uniform(0.5, 2.0)
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 / (1.0 + aspect)
        
        # Estimate radius
        r_est = np.sqrt(w * h / (N * np.pi)) * 0.9
        
        x = np.zeros(3 * N)
        for i in range(N):
            x[i*3] = np.random.uniform(r_est, w - r_est)
            x[i*3+1] = np.random.uniform(r_est, h - r_est)
            x[i*3+2] = r_est * np.random.uniform(0.8, 1.2)
        
        T = T0
        current_energy = _energy(x, aspect)
        current_score = _get_score(x, aspect)
        
        best_x_local = x.copy()
        best_score_local = current_score
        best_aspect_local = aspect
        
        for it in range(max_iter):
            if time.time() > deadline - 10.0:
                break
                
            # Propose move
            x_new = x.copy()
            
            # Perturb positions
            x_new[0::3] += np.random.normal(0, 0.05 * T, N)
            x_new[1::3] += np.random.normal(0, 0.05 * T, N)
            
            # Perturb radii
            x_new[2::3] += np.random.normal(0, 0.01 * T, N)
            
            # Clamp radii to positive
            x_new[2::3] = np.maximum(x_new[2::3], 1e-4)
            
            # Occasionally perturb aspect
            if random.random() < 0.01:
                aspect_new = aspect * np.exp(np.random.normal(0, 0.05))
                aspect_new = np.clip(aspect_new, 0.3, 3.0)
            else:
                aspect_new = aspect
                
            w_new = 2.0 * aspect_new / (1.0 + aspect_new)
            h_new = 2.0 / (1.0 + aspect_new)
            
            # Clamp positions to new bounds roughly to avoid huge penalties
            x_new[0::3] = np.clip(x_new[0::3], 0, w_new)
            x_new[1::3] = np.clip(x_new[1::3], 0, h_new)
            
            new_energy = _energy(x_new, aspect_new)
            
            # Acceptance criterion
            if new_energy < current_energy:
                x = x_new
                aspect = aspect_new
                current_energy = new_energy
            else:
                delta_e = new_energy - current_energy
                if random.random() < np.exp(-delta_e / T):
                    x = x_new
                    aspect = aspect_new
                    current_energy = new_energy
            
            T *= alpha
            if T < T_end:
                break
            
            # Update local best
            score_new = _get_score(x, aspect)
            if score_new > best_score_local:
                best_score_local = score_new
                best_x_local = x.copy()
                best_aspect_local = aspect
        
        # Polish the best local solution
        if best_score_local > -np.inf:
            x_polished = _polish(best_x_local, best_aspect_local, time_limit=5.0)
            score_polished = _get_score(x_polished, best_aspect_local)
            
            if score_polished > best_score_local:
                best_score_local = score_polished
                best_x_local = x_polished
            
            if best_score_local > best_score:
                best_score = best_score_local
                best_x = best_x_local
                best_aspect = best_aspect_local
        
        # Early stop if we found a very good solution?
        # Theoretical max sum of radii for 21 circles in perimeter 4 (square 1x1)
        # Area = 1. 21 * pi * r^2 <= 1 => r <= sqrt(1/(21*pi)) ~ 0.123
        # Sum <= 21 * 0.123 ~ 2.58. 
        # Realistically, packing efficiency ~ 0.8-0.9. 
        # If score > 2.2, it's quite good.
        if best_score > 2.2:
            break

    if best_x is None:
        # Fallback
        w = 1.0
        h = 1.0
        r = 0.08
        res = np.zeros((N, 3))
        for i in range(N):
            row = i // 5
            col = i % 5
            res[i, 0] = (col + 0.5) * 0.2
            res[i, 1] = (row + 0.5) * 0.2
            res[i, 2] = r
        return res

    # Final check and return
    if not _check_validity(best_x, best_aspect):
        # If invalid, try to fix by scaling down radii slightly
        r = best_x[2::3]
        r *= 0.95
        best_x[2::3] = r
        if not _check_validity(best_x, best_aspect):
             # Return fallback
             w = 1.0
             h = 1.0
             r = 0.08
             res = np.zeros((N, 3))
             for i in range(N):
                 row = i // 5
                 col = i % 5
                 res[i, 0] = (col + 0.5) * 0.2
                 res[i, 1] = (row + 0.5) * 0.2
                 res[i, 2] = r
             return res

    return np.column_stack((best_x[0::3], best_x[1::3], best_x[2::3]))

# EVOLVE-BLOCK-END
