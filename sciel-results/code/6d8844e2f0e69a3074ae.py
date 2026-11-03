"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import least_squares
from scipy.spatial import Voronoi

def _make_barrier_resids(x, y, r, beta):
    """
    Construct residuals for least_squares.
    Residuals are:
    1. -beta * r_i for each circle (to maximize sum of r)
    2. Barrier terms for wall violations: max(0, r_i - x_i), max(0, r_i - (1-x_i)), etc.
    3. Barrier terms for overlap violations: max(0, r_i + r_j - dist_ij)
    """
    n = len(r)
    resids = []
    
    # Objective: maximize sum(r) -> minimize -sum(r)
    # We scale this by beta to balance with barriers
    resids.extend([-beta * r_i for r_i in r])
    
    # Wall constraints:
    # x_i >= r_i  -> r_i - x_i <= 0
    # 1-x_i >= r_i -> r_i - (1-x_i) <= 0
    # y_i >= r_i  -> r_i - y_i <= 0
    # 1-y_i >= r_i -> r_i - (1-y_i) <= 0
    for i in range(n):
        resids.append(max(0.0, r[i] - x[i]))
        resids.append(max(0.0, r[i] - (1.0 - x[i])))
        resids.append(max(0.0, r[i] - y[i]))
        resids.append(max(0.0, r[i] - (1.0 - y[i])))
        
    # Overlap constraints:
    # dist_ij >= r_i + r_j -> r_i + r_j - dist_ij <= 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            violation = r[i] + r[j] - dist
            if violation > 0:
                resids.append(violation)
                
    return np.array(resids)

def _smooth_packing(x0, y0, r0, deadline, beta=10.0, max_nfev=2000):
    """
    Run least_squares to refine a packing.
    """
    n = len(r0)
    # Flatten state
    def flatten(xy, r):
        return np.concatenate([xy.ravel(), r])
        
    def unflatten(z):
        xy = z[:2*n].reshape(n, 2)
        r = z[2*n:]
        return xy, r
        
    def residuals(z):
        xy, r = unflatten(z)
        # Ensure r is positive for stability in residual calc, though optimizer may go negative
        r_safe = np.maximum(r, 1e-8)
        return _make_barrier_resids(xy[:,0], xy[:,1], r_safe, beta)
        
    def jac(z):
        # Numerical Jacobian is often sufficient and robust for this size
        return None 
        
    z0 = flatten(x0, y0, r0)
    
    try:
        res = least_squares(
            residuals, 
            z0, 
            method='lm', # Levenberg-Marquardt is good for this structure
            max_nfev=max_nfev,
            # bounds=([0]*2*n + [0]*n, [1]*2*n + [1]*n) # LM doesn't support bounds well in older scipy, or we handle via barriers
        )
    except Exception:
        return x0, y0, r0, 0.0, False
        
    xy_opt, r_opt = unflatten(res.x)
    
    # Post-process: Clip to valid range
    for i in range(n):
        r_opt[i] = max(0.0, r_opt[i])
        r_opt[i] = min(r_opt[i], 0.5) # Max possible radius
        x_opt = np.clip(xy_opt[i,0], r_opt[i], 1.0 - r_opt[i])
        y_opt = np.clip(xy_opt[i,1], r_opt[i], 1.0 - r_opt[i])
        xy_opt[i,0] = x_opt
        xy_opt[i,1] = y_opt
        
    # Check validity
    valid = True
    for i in range(n):
        if xy_opt[i,0] - r_opt[i] < -1e-6 or 1.0 - xy_opt[i,0] - r_opt[i] < -1e-6:
            valid = False
        if xy_opt[i,1] - r_opt[i] < -1e-6 or 1.0 - xy_opt[i,1] - r_opt[i] < -1e-6:
            valid = False
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(xy_opt[i] - xy_opt[j])
            if d < r_opt[i] + r_opt[j] - 1e-6:
                valid = False
                
    total = np.sum(r_opt)
    return xy_opt, xy_opt[:,1], r_opt, total, valid

def _init_grid(n, seed=0):
    side = int(np.ceil(np.sqrt(n)))
    xs, ys = np.mgrid[0:side, 0:side]
    xs = xs.ravel()[:n]
    ys = ys.ravel()[:n]
    x = (xs + 0.5) / side
    y = (ys + 0.5) / side
    r = np.full(n, 0.5/side - 0.01)
    return x, y, r

def _init_hex(n, seed=0):
    # Hexagonal packing initialization
    rng = np.random.RandomState(seed)
    x = np.zeros(n)
    y = np.zeros(n)
    r = np.zeros(n)
    
    # Start with a few large circles and fill in
    # Simple approach: random points, then push apart
    x = rng.uniform(0.2, 0.8, n)
    y = rng.uniform(0.2, 0.8, n)
    r = np.full(n, 0.05)
    return x, y, r

def _init_voronoi(n, seed=0):
    # Generate random points, compute Voronoi, set radius to min distance to edge
    rng = np.random.RandomState(seed)
    x = rng.uniform(0.1, 0.9, n)
    y = rng.uniform(0.1, 0.9, n)
    
    # Add boundary points to define Voronoi cells correctly
    bx = np.concatenate([np.linspace(0, 1, 10), np.zeros(10), np.ones(10)])
    by = np.concatenate([np.zeros(10), np.linspace(0, 1, 10), np.ones(10)])
    # Actually, easier to just use distance to boundary and nearest neighbor
    r = np.zeros(n)
    for i in range(n):
        # Distance to walls
        d_wall = min(x[i], 1-x[i], y[i], 1-y[i])
        # Distance to others
        d_min = d_wall
        for j in range(n):
            if i == j: continue
            d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            d_min = min(d_min, d/2)
        r[i] = d_min * 0.95 # Slightly shrink to allow optimization
        
    return x, y, r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_total = 0.0
    best_x = None
    best_y = None
    best_r = None
    best_valid = False
    
    # Strategies:
    # 1. Grid init + LM
    # 2. Voronoi init + LM
    # 3. Random init + LM
    
    initializers = [
        ("grid", _init_grid),
        ("voronoi", _init_voronoi),
        ("hex", _init_hex),
    ]
    
    # We will run multiple seeds for each initializer
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    # To manage time, we interleave
    # First pass: quick run with low max_nfev
    # Second pass: refine best with high max_nfev
    
    candidates = []
    
    for seed in seeds:
        if time.time() > deadline - 20: break
        
        for name, init_func in initializers:
            if time.time() > deadline - 20: break
            
            try:
                x, y, r = init_func(N, seed=seed)
                # Quick optimization
                x_o, y_o, r_o, total, valid = _smooth_packing(x, y, r, deadline, beta=5.0, max_nfev=500)
                if valid and total > best_total:
                    best_total = total
                    best_valid = True
                    best_x = x_o.copy()
                    best_y = y_o.copy()
                    best_r = r_o.copy()
                candidates.append((total, x_o, y_o, r_o, valid))
            except Exception:
                pass
                
    # Refine top candidates with higher precision
    # Sort by total
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Take top 5 unique-ish candidates
    top_candidates = []
    seen = set()
    for c in candidates:
        if c[4]: # valid
            # Simple dedup by rounded positions
            key = (round(c[1][0],2), round(c[1][1],2))
            if key not in seen:
                seen.add(key)
                top_candidates.append(c)
        if len(top_candidates) >= 5:
            break
            
    for c in top_candidates:
        if time.time() > deadline - 5: break
        total, x, y, r, valid = c
        try:
            # Higher beta for tighter constraints, more iterations
            x_o, y_o, r_o, total_o, valid_o = _smooth_packing(x, y, r, deadline, beta=20.0, max_nfev=2000)
            if valid_o and total_o > best_total:
                best_total = total_o
                best_valid = True
                best_x = x_o.copy()
                best_y = y_o.copy()
                best_r = r_o.copy()
        except Exception:
            pass
            
    # Final check and formatting
    if best_x is None or not best_valid:
        # Fallback: simple grid
        x, y, r = _init_grid(N, seed=0)
        best_x = x
        best_y = y
        best_r = r
        best_valid = True
        best_total = np.sum(r)
        
    # Ensure strict validity for final output
    # If invalid, shrink radii slightly
    for i in range(N):
        r_i = best_r[i]
        x_i = best_x[i]
        y_i = best_y[i]
        # Check walls
        max_r = min(x_i, 1-x_i, y_i, 1-y_i)
        if r_i > max_r:
            r_i = max_r * 0.99
        # Check overlaps
        for j in range(i+1, N):
            d = np.sqrt((best_x[i]-best_x[j])**2 + (best_y[i]-best_y[j])**2)
            max_r_overlap = (d - best_r[j]) * 0.99
            if r_i > max_r_overlap:
                r_i = max_r_overlap
        best_r[i] = max(0.0, r_i)
        
    # Re-validate after shrinking
    valid = True
    for i in range(N):
        if best_x[i] - best_r[i] < -1e-7 or 1.0 - best_x[i] - best_r[i] < -1e-7:
            valid = False
        if best_y[i] - best_r[i] < -1e-7 or 1.0 - best_y[i] - best_r[i] < -1e-7:
            valid = False
    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt((best_x[i]-best_x[j])**2 + (best_y[i]-best_y[j])**2)
            if d < best_r[i] + best_r[j] - 1e-7:
                valid = False
                
    if not valid:
        # Extreme fallback: tiny circles in grid
        side = 6
        for i in range(N):
            gx = i % side
            gy = i // side
            best_x[i] = (gx + 0.5) / side
            best_y[i] = (gy + 0.5) / side
            best_r[i] = 0.5/side - 0.05
            
    result = np.zeros((N, 3))
    result[:, 0] = best_x
    result[:, 1] = best_y
    result[:, 2] = best_r
    
    return result
# EVOLVE-BLOCK-END
