"""Circle packing in perimeter-4 rectangle: Unified SLSQP optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
# Safety margin: The evaluator kills at 120s (or 150s in some contexts, prompt says 120s).
# We set a conservative deadline to ensure we return a valid result.
DEADLINE = 115.0 
_START_TIME = None

def _objective(x, N):
    """
    Minimize negative sum of radii.
    x layout: [w, r_0..r_{N-1}, cx_0..cx_{N-1}, cy_0..cy_{N-1}]
    """
    w = x[0]
    r = x[1:1+N]
    return -np.sum(r)

def _make_constraints(w_initial, N):
    """
    Create constraint functions for SLSQP.
    Constraints are of the form fun(x) >= 0.
    """
    cons = []
    
    # 1. Rectangle width w must be reasonable (e.g., 0.2 to 1.8)
    # w > 0.2
    cons.append({'type': 'ineq', 'fun': lambda x: x[0] - 0.2})
    # w < 1.8 (since h = 2-w, h > 0.2)
    cons.append({'type': 'ineq', 'fun': lambda x: 1.8 - x[0]})
    
    # 2. Radii must be positive
    for i in range(N):
        idx = 1 + i
        cons.append({'type': 'ineq', 'fun': lambda x, idx=idx: x[idx] - 1e-4})
        
    # 3. Boundary constraints
    # For each circle i:
    # cx_i >= r_i  =>  x[N+1+i] - x[1+i] >= 0
    # w - cx_i >= r_i =>  x[0] - x[N+1+i] - x[1+i] >= 0
    # cy_i >= r_i  =>  x[2N+1+i] - x[1+i] >= 0
    # h - cy_i >= r_i =>  (2 - x[0]) - x[2N+1+i] - x[1+i] >= 0
    
    for i in range(N):
        r_idx = 1 + i
        cx_idx = N + 1 + i
        cy_idx = 2 * N + 1 + i
        
        # Left boundary
        cons.append({'type': 'ineq', 'fun': lambda x, cx=cx_idx, r=r_idx: x[cx] - x[r]})
        # Right boundary
        cons.append({'type': 'ineq', 'fun': lambda x, w=0, cx=cx_idx, r=r_idx: x[w] - x[cx] - x[r]})
        # Bottom boundary
        cons.append({'type': 'ineq', 'fun': lambda x, cy=cy_idx, r=r_idx: x[cy] - x[r]})
        # Top boundary (h = 2 - w)
        cons.append({'type': 'ineq', 'fun': lambda x, w=0, cy=cy_idx, r=r_idx: (2.0 - x[w]) - x[cy] - x[r]})
        
    # 4. Non-overlap constraints
    # dist(i,j)^2 >= (r_i + r_j)^2
    # (cx_i - cx_j)^2 + (cy_i - cy_j)^2 - (r_i + r_j)^2 >= 0
    for i in range(N):
        for j in range(i + 1, N):
            ri_idx = 1 + i
            rj_idx = 1 + j
            cxi_idx = N + 1 + i
            cxj_idx = N + 1 + j
            cyi_idx = 2 * N + 1 + i
            cyj_idx = 2 * N + 1 + j
            
            def non_overlap(x, ri=ri_idx, rj=rj_idx, cxi=cxi_idx, cxj=cxj_idx, cyi=cyi_idx, cyj=cyj_idx):
                dx = x[cxi] - x[cxj]
                dy = x[cyi] - x[cyj]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (x[ri] + x[rj])**2
                return dist_sq - min_dist_sq
            
            cons.append({'type': 'ineq', 'fun': non_overlap})
            
    return cons

def _grid_seed(w, h, rows, cols, N):
    """
    Generate an initial feasible guess for circles in a grid.
    Returns x array: [w, r..., cx..., cy...]
    """
    x = np.zeros(3 * N + 1)
    x[0] = w
    
    # Calculate cell size
    cw = w / cols
    ch = h / rows
    
    # Initial radius estimate: slightly less than half the min cell dimension
    r_init = 0.45 * min(cw, ch)
    
    count = 0
    for i in range(rows):
        for j in range(cols):
            if count >= N:
                break
            cx = (j + 0.5) * cw
            cy = (i + 0.5) * ch
            r = r_init
            
            # Ensure it fits in the cell (simple check)
            if cx < r or cx > w - r or cy < r or cy > h - r:
                r = min(cx, w-cx, cy, h-cy) * 0.99
                if r < 1e-4:
                    r = 1e-4
                    cx = w/2
                    cy = h/2
            
            x[1 + count] = r
            x[N + 1 + count] = cx
            x[2 * N + 1 + count] = cy
            count += 1
            
    return x

def _is_feasible(x, tol=1e-5):
    """Check if the solution is valid."""
    w = x[0]
    h = 2.0 - w
    r = x[1:1+N]
    cx = x[N+1:2*N+1]
    cy = x[2*N+1:3*N+1]
    
    if w <= 0.01 or h <= 0.01:
        return False
    if np.any(r < -tol):
        return False
        
    # Boundaries
    if np.any(cx < r - tol) or np.any(cx > w - r + tol):
        return False
    if np.any(cy < r - tol) or np.any(cy > h - r + tol):
        return False
        
    # Overlaps
    for i in range(N):
        for j in range(i+1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = r[i] + r[j]
            if dist < min_dist - tol:
                return False
    return True

def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    best_val = -np.inf
    best_result = None
    
    # Define a set of initial configurations to try.
    # We vary the aspect ratio and the grid layout to explore different local optima.
    # The optimizer will adjust w, so the initial w is just a seed.
    
    configs = [
        # (initial_w, rows, cols)
        (1.0, 5, 5),    # Square, 5x5 grid (25 cells, use 21)
        (1.0, 4, 6),    # Square, 4x6 grid
        (1.2, 4, 6),    # Wider
        (0.8, 6, 4),    # Taller
        (1.0, 3, 7),    # Wide and short rows
        (1.0, 7, 3),    # Tall and narrow cols
        (1.1, 5, 5),
        (0.9, 5, 5),
        (1.3, 4, 6),
        (0.7, 6, 4),
    ]
    
    constraints_cache = {}
    
    for w_init, rows, cols in configs:
        if time.time() - _START_TIME > DEADLINE - 5:
            break
            
        h_init = 2.0 - w_init
        
        # Generate seed
        x0 = _grid_seed(w_init, h_init, rows, cols, N)
        
        # Bounds for SLSQP
        # w: [0.2, 1.8]
        # r: [1e-4, 0.9] (max radius approx half of min(w,h) which is <= 1)
        # cx: [0, 2]
        # cy: [0, 2]
        bounds = [(0.2, 1.8)] # w
        for i in range(N):
            bounds.append((1e-4, 0.9)) # r
        for i in range(N):
            bounds.append((0.0, 2.0)) # cx
        for i in range(N):
            bounds.append((0.0, 2.0)) # cy
            
        # Create constraints. Note: SLSQP constraints depend on w (x[0]) dynamically,
        # so we don't need to recreate them for every w change, the lambda closures handle it.
        # However, the number of constraints is fixed.
        if 'cons' not in constraints_cache:
            constraints_cache['cons'] = _make_constraints(w_init, N)
        
        cons = constraints_cache['cons']
        
        try:
            res = minimize(
                lambda x: _objective(x, N),
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 200, 'ftol': 1e-8, 'disp': False}
            )
            
            x_opt = res.x
            
            # Verify feasibility
            if _is_feasible(x_opt, tol=1e-4):
                val = np.sum(x_opt[1:1+N])
                if val > best_val:
                    best_val = val
                    best_result = x_opt.copy()
                    
        except Exception as e:
            # print(f"Optimization failed for config {w_init}, {rows}, {cols}: {e}")
            pass
        
        # If we have a good result, we might stop early if time is tight
        if best_val > 2.5: # Heuristic: 21 circles, avg radius ~0.12 -> sum ~2.5. If we get high, we're good.
             # Actually, max possible sum is bounded by area. Area=1. 
             # 21 circles of radius r. 21 * pi * r^2 <= 1 => r <= sqrt(1/(21*pi)) ~ 0.123.
             # Sum <= 21 * 0.123 ~ 2.58.
             # So 2.5 is very good.
             if time.time() - _START_TIME > DEADLINE - 20:
                 break

    # Fallback: If no valid result found, return a trivial valid packing
    if best_result is None:
        # Simple grid in 1x1 square
        w = 1.0
        h = 1.0
        # 5x5 grid
        side = 5
        r = 0.19 # slightly less than 0.2
        pts = []
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                x = (j + 0.5) * (w / side)
                y = (i + 0.5) * (h / side)
                pts.append([x, y, r])
                count += 1
        
        # Ensure it's valid (it should be for 5x5 with r=0.19)
        # dist between centers = 0.2. 2*r = 0.38 > 0.2. Overlap!
        # Need r <= 0.1.
        r = 0.09
        pts = []
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                x = (j + 0.5) * (w / side)
                y = (i + 0.5) * (h / side)
                pts.append([x, y, r])
                count += 1
        return np.array(pts)

    x_opt = best_result
    w = x_opt[0]
    h = 2.0 - w
    r = x_opt[1:1+N]
    cx = x_opt[N+1:2*N+1]
    cy = x_opt[2*N+1:3*N+1]
    
    # Final safety clamp: ensure strict validity for the evaluator
    # Reduce radii slightly if needed to handle numerical drift
    for i in range(N):
        # Boundary check
        r_i = r[i]
        cx_i = cx[i]
        cy_i = cy[i]
        
        # Max r allowed by boundaries
        max_r = min(cx_i, w - cx_i, cy_i, h - cy_i)
        if r_i > max_r:
            r[i] = max_r * 0.999
            
    # Overlap check and shrink
    for _ in range(10):
        overlap = False
        for i in range(N):
            for j in range(i+1, N):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    # Shrink both
                    reduce = (min_dist - dist) / 2.0 + 0.001
                    r[i] = max(r[i] - reduce, 1e-4)
                    r[j] = max(r[j] - reduce, 1e-4)
                    overlap = True
        if not overlap:
            break
            
    # Re-check boundaries after shrink (shrink only helps boundaries, so no need to re-clamp positions, 
    # but positions might have drifted out of bounds due to optimizer noise)
    for i in range(N):
        cx[i] = np.clip(cx[i], r[i], w - r[i])
        cy[i] = np.clip(cy[i], r[i], h - r[i])
        
    return np.column_stack((cx, cy, r))
