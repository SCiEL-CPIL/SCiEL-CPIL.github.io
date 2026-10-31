"""Circle packing in perimeter-4 rectangle: Unified SLSQP optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
PERIMETER = 4.0
# Safety margin: The evaluator kills at 120s (or 150s in some contexts).
# We set a conservative deadline to ensure we return a valid result.
DEADLINE = 110.0 
_START_TIME = None

def _objective(x):
    """
    Minimize negative sum of radii.
    x layout: [w, r_0..r_{N-1}, cx_0..cx_{N-1}, cy_0..cy_{N-1}]
    """
    r = x[1:1+N]
    return -np.sum(r)

def _make_constraints(N):
    """
    Create constraint functions for SLSQP.
    Constraints are of the form fun(x) >= 0.
    x layout: [w, r_0..r_{N-1}, cx_0..cx_{N-1}, cy_0..cy_{N-1}]
    Indices:
    w: 0
    r_i: 1+i
    cx_i: N+1+i
    cy_i: 2N+1+i
    """
    cons = []
    
    # 1. Rectangle width w must be reasonable. 
    # Perimeter = 2(w+h) = 4 => w+h=2. 
    # To avoid degenerate rectangles, keep w in [0.3, 1.7] => h in [0.3, 1.7]
    cons.append({'type': 'ineq', 'fun': lambda x: x[0] - 0.3})
    cons.append({'type': 'ineq', 'fun': lambda x: 1.7 - x[0]})
    
    # 2. Radii must be positive (lower bound handled by bounds, but explicit constraint helps)
    # We rely on bounds for r > 0 mostly, but SLSQP handles bounds well.
    
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

def _refine_local(x, N, max_iter=50):
    """
    Simple local refinement: try to increase radii by moving circles slightly.
    This is a heuristic to squeeze out a bit more value after SLSQP.
    """
    # We skip complex local search here to save time, SLSQP is usually good enough
    # with multiple restarts. If we have time, we could do a gradient step on radii only.
    return x

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
        (1.0, 6, 4),
        (1.0, 4, 5),
    ]
    
    constraints_cache = _make_constraints(N)
    
    for w_init, rows, cols in configs:
        if time.time() - _START_TIME > DEADLINE:
            break
            
        h_init = 2.0 - w_init
        
        # Generate seed
        x0 = _grid_seed(w_init, h_init, rows, cols, N)
        
        # Bounds for SLSQP
        # w: [0.3, 1.7]
        # r: [1e-4, 0.9] (max radius approx half of min(w,h) which is <= 1)
        # cx: [0, 2]
        # cy: [0, 2]
        bounds = [(0.3, 1.7)] # w
        for i in range(N):
            bounds.append((1e-4, 0.9)) # r
        for i in range(N):
            bounds.append((0.0, 2.0)) # cx
        for i in range(N):
            bounds.append((0.0, 2.0)) # cy
            
        try:
            res = minimize(
                _objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_cache,
                options={'maxiter': 200, 'ftol': 1e-8, 'disp': False}
            )
            
            x_opt = res.x
            
            # Verify feasibility
            if _is_feasible(x_opt, tol=1e-4):
                val = np.sum(x_opt[1:1+N])
                if val > best_val:
                    best_val = val
                    best_result = x_opt.copy()
                    
        except Exception:
            pass
        
        # If we have a good result, we might stop early if time is tight
        if best_val > 2.5: # Heuristic: 21 circles, avg radius ~0.12 -> sum ~2.5. If we get high, we're good.
             if time.time() - _START_TIME > DEADLINE * 0.8:
                 break

    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        w = 1.0
        h = 1.0
        cells = []
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                cx = (j + 0.5) / side
                cy = (i + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append((cx, cy, r))
                count += 1
        
        # Convert to x array format [w, r..., cx..., cy...]
        x = np.zeros(3 * N + 1)
        x[0] = w
        for i, (cx, cy, r) in enumerate(cells):
            x[1+i] = r
            x[N+1+i] = cx
            x[2*N+1+i] = cy
        best_result = x
        best_val = np.sum(x[1:1+N])

    # Extract circles from best_result
    w = best_result[0]
    h = 2.0 - w
    r = best_result[1:1+N]
    cx = best_result[N+1:2*N+1]
    cy = best_result[2*N+1:3*N+1]
    
    circles = np.column_stack((cx, cy, r))
    
    return circles

if __name__ == "__main__":
    res = construct()
    print(res.shape)
    print(res[:3])
