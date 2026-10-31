"""
Circle packing in perimeter-4 rectangle.
Strategy:
1. Use a structural change: instead of optimizing all radii and positions simultaneously with a soft penalty,
   we fix the rectangle aspect ratio (or optimize it coarsely) and use a "contact graph" inspired initialization.
   However, the directive asks for a structural change like Delaunay/contact graph.
   A simpler and more robust structural change for this specific problem (N=21) is to use a 
   "rigid body" approach or a better local search.
   
   Let's look at the inspiration: SLSQP with explicit constraints. This is generally more robust than L-BFGS-B with penalties 
   for hard constraints like non-overlap. The penalty method often struggles with the non-differentiability of the max(0, viol) 
   or the quadratic penalty scaling. SLSQP handles the KKT conditions for inequality constraints directly.
   
   Another key idea from inspiration: Grid-based initialization tailored to aspect ratio.
   
   Let's combine:
   - Use SLSQP with explicit inequality constraints (from inspiration 1).
   - Use a multi-start strategy with diverse aspect ratios (from inspiration 1).
   - Add a "repair" step: if SLSQP fails or returns invalid, try to shrink radii slightly to make it valid, or discard.
   - Optimize W (width) as a variable? The inspiration fixes W. Optimizing W adds 1 variable. 
     For N=21, 3N+1 = 64 variables. SLSQP can handle this but might be slow.
     Let's fix W to a few good values (1.0, 1.2, 0.8, etc.) to keep the problem smaller (3N = 63 vars) and faster per run, 
     allowing more restarts. The optimal aspect ratio for 21 circles is likely close to square or slightly rectangular.
     
   Actually, let's allow W to vary but use a coarser approach:
   Loop over a few aspect ratios. For each, run SLSQP.
   
   To improve on the seed (score 1.0), we need to beat 2.079.
   The theoretical max for 21 circles in a square of side 1 (perimeter 4) is roughly... 
   Area of 21 circles = 21 * pi * r^2. Area of square = 1.
   If they fit perfectly, r ~ sqrt(1/(21*pi)) ~ 0.123. Sum = 21 * 0.123 ~ 2.58.
   But packing efficiency is ~0.90-0.95. So sum ~ 2.3-2.4 is plausible.
   
   Let's implement the SLSQP approach with multiple aspect ratios and restarts.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 21
DEADLINE = 115.0  # Safety margin before 120s kill

def _validate(x, y, r, w, h, tol=1e-6):
    """Check if the packing is valid."""
    if np.any(r < 0):
        return False
    if np.any(x < r - tol) or np.any(x > w - r + tol):
        return False
    if np.any(y < r - tol) or np.any(y > h - r + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            if dist < r[i] + r[j] - tol:
                return False
    return True

def _solve_for_aspect(w, h, restart_idx, start_time):
    """
    Solve for a fixed rectangle w x h (w+h=2).
    Uses SLSQP with explicit constraints.
    """
    if time.time() - start_time > DEADLINE - 10:
        return None, None
    
    # Initialize
    rng = np.random.RandomState(42 + restart_idx)
    
    # Grid initialization
    # Estimate grid dimensions
    cols = int(np.sqrt(N * w / h))
    if cols < 2: cols = 2
    rows = int(np.ceil(N / cols))
    
    # Adjust grid to fit N
    while cols * rows < N:
        if cols <= rows:
            cols += 1
        else:
            rows += 1
            
    positions = np.zeros((N, 2))
    radii = np.zeros(N)
    
    idx = 0
    for r_idx in range(rows):
        for c_idx in range(cols):
            if idx >= N:
                break
            x = (c_idx + 0.5) * w / cols
            y = (r_idx + 0.5) * h / rows
            
            # Add jitter for different restarts
            if restart_idx > 0:
                jitter = 0.1 * min(w/cols, h/rows)
                x += rng.uniform(-jitter, jitter)
                y += rng.uniform(-jitter, jitter)
            
            x = np.clip(x, 0.01, w - 0.01)
            y = np.clip(y, 0.01, h - 0.01)
            
            positions[idx] = [x, y]
            idx += 1
            
    # If we didn't fill all, fill remaining randomly (shouldn't happen with while loop)
    while idx < N:
        positions[idx] = [rng.uniform(0.1, w-0.1), rng.uniform(0.1, h-0.1)]
        idx += 1
        
    # Calculate initial radii
    # Min distance to neighbors and walls
    min_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt((positions[i,0]-positions[j,0])**2 + (positions[i,1]-positions[j,1])**2)
            min_dist = min(min_dist, d)
    for i in range(N):
        min_dist = min(min_dist, positions[i,0], w-positions[i,0], positions[i,1], h-positions[i,1])
        
    r0 = 0.4 * min_dist
    radii = np.full(N, r0)
    
    # Flatten variables: x1, y1, r1, x2, y2, r2, ...
    x0 = np.zeros(3*N)
    for i in range(N):
        x0[3*i] = positions[i, 0]
        x0[3*i+1] = positions[i, 1]
        x0[3*i+2] = radii[i]
        
    # Objective: Maximize sum(r) => Minimize -sum(r)
    def objective(vars_):
        r = vars_[2::3]
        return -np.sum(r)
        
    # Constraints
    def cons_inside(vars_):
        x = vars_[0::3]
        y = vars_[1::3]
        r = vars_[2::3]
        return np.concatenate([
            x - r,
            w - x - r,
            y - r,
            h - y - r
        ])
        
    def cons_non_overlap(vars_):
        x = vars_[0::3]
        y = vars_[1::3]
        r = vars_[2::3]
        cons = []
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                cons.append(dist - r[i] - r[j])
        return np.array(cons)
        
    def cons_positive_r(vars_):
        r = vars_[2::3]
        return r
        
    constraints = [
        {'type': 'ineq', 'fun': cons_inside},
        {'type': 'ineq', 'fun': cons_non_overlap},
        {'type': 'ineq', 'fun': cons_positive_r}
    ]
    
    try:
        result = minimize(
            objective, x0, method='SLSQP',
            constraints=constraints,
            options={'maxiter': 100, 'ftol': 1e-8}
        )
        
        x_res = result.x[0::3]
        y_res = result.x[1::3]
        r_res = result.x[2::3]
        
        # Validation
        if _validate(x_res, y_res, r_res, w, h):
            score = np.sum(r_res)
            return score, np.column_stack((x_res, y_res, r_res))
        else:
            # Try to repair by shrinking radii slightly
            # This is a heuristic repair
            scale = 0.95
            r_res_fixed = r_res * scale
            if _validate(x_res, y_res, r_res_fixed, w, h):
                score = np.sum(r_res_fixed)
                return score, np.column_stack((x_res, y_res, r_res_fixed))
            return None, None
            
    except Exception:
        return None, None

def construct():
    start_time = time.time()
    best_score = -np.inf
    best_result = None
    
    # Aspect ratios to try
    # w + h = 2
    # Square: w=1, h=1
    # Rect: w=1.2, h=0.8; w=0.8, h=1.2
    aspect_configs = [
        (1.0, 1.0),
        (1.2, 0.8),
        (0.8, 1.2),
        (1.4, 0.6),
        (0.6, 1.4),
        (1.1, 0.9),
        (0.9, 1.1)
    ]
    
    max_restarts_per_aspect = 3
    
    for w, h in aspect_configs:
        if time.time() - start_time > DEADLINE - 15:
            break
            
        for restart in range(max_restarts_per_aspect):
            if time.time() - start_time > DEADLINE - 5:
                break
                
            score, result = _solve_for_aspect(w, h, restart, start_time)
            
            if score is not None and score > best_score:
                best_score = score
                best_result = result
                
            # Early exit if we find a very good solution
            if best_score > 2.4:
                break
                
    # Fallback: Simple grid packing
    if best_result is None or best_score < 2.0:
        # 5x5 grid in 1x1 square
        side = 5
        r = 0.5 / side - 1e-3
        cells = []
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                cells.append([(i + 0.5) / side, (j + 0.5) / side, r])
                count += 1
        best_result = np.array(cells)
        best_score = np.sum(best_result[:, 2])
        
    return best_result

# EVOLVE-BLOCK-END
