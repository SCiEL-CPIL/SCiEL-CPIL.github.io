import numpy as np
import time
import math
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START

def _validate_and_compute(radii, x, y, W, H):
    """Check validity and compute sum of radii. Returns (is_valid, value)."""
    n = len(radii)
    # Check bounds
    for i in range(n):
        if x[i] < radii[i] - 1e-9 or x[i] > W - radii[i] + 1e-9:
            return False, 0.0
        if y[i] < radii[i] - 1e-9 or y[i] > H - radii[i] + 1e-9:
            return False, 0.0
    # Check pairwise distances
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx*dx + dy*dy
            min_dist = radii[i] + radii[j]
            if dist_sq < min_dist*min_dist - 1e-9:
                return False, 0.0
    # Check perimeter
    perim = 2*(W + H)
    if perim > 4.0 + 1e-9:
        return False, 0.0
    value = np.sum(radii)
    return True, value

def _penalty_objective(params, N):
    """
    Objective function for scipy.optimize.minimize.
    params: [W, x_1, y_1, r_1, ..., x_N, y_N, r_N]
    """
    W = params[0]
    H = 2.0 - W
    
    if W <= 0.1 or H <= 0.1:
        return 1e6

    x = params[1::3]
    y = params[2::3]
    r = params[3::3]
    
    # Objective: -sum(r) + penalties
    obj = -np.sum(r)
    
    # Boundary penalties
    # x < r or x > W-r
    bound_x_low = np.maximum(0, r - x)
    bound_x_high = np.maximum(0, x - (W - r))
    bound_y_low = np.maximum(0, r - y)
    bound_y_high = np.maximum(0, y - (H - r))
    
    penalty_bound = np.sum(bound_x_low**2) + np.sum(bound_x_high**2) + \
                    np.sum(bound_y_low**2) + np.sum(bound_y_high**2)
    
    # Overlap penalties
    # Vectorized distance calculation
    # d_ij^2 = (x_i - x_j)^2 + (y_i - y_j)^2
    # We want max(0, r_i + r_j - d_ij)^2
    # For efficiency, loop is fine for N=21 in Python if called many times? 
    # No, scipy calls this many times. Vectorization is better.
    
    # Create distance matrix
    # x: (N,), y: (N,)
    # dx: (N, N)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist_sq = dx**2 + dy**2
    dist = np.sqrt(dist_sq + 1e-12)
    
    # r_i + r_j
    r_sum = r[:, None] + r[None, :]
    
    # Overlap amount
    overlap = np.maximum(0, r_sum - dist)
    # Zero out diagonal (i=j)
    np.fill_diagonal(overlap, 0)
    
    penalty_overlap = np.sum(overlap**2)
    
    # Weights
    w_bound = 100.0
    w_overlap = 100.0
    
    return obj + w_bound * penalty_bound + w_overlap * penalty_overlap

def _unpack(params):
    W = params[0]
    x = params[1::3]
    y = params[2::3]
    r = params[3::3]
    return W, x, y, r

def _pack(W, x, y, r):
    return np.concatenate(([W], x, y, r)).flatten()

def _seed_grid_3x7(W, H):
    """Create a 3x7 grid of circles."""
    x = []
    y = []
    r = []
    rows = 3
    cols = 7
    for i in range(rows):
        for j in range(cols):
            cx = (j + 0.5) * W / cols
            cy = (i + 0.5) * H / rows
            rad = 0.5 * min(W/cols, H/rows) * 0.95
            x.append(cx)
            y.append(cy)
            r.append(rad)
    return np.array(x), np.array(y), np.array(r)

def _seed_grid_5x5_subset(W, H):
    """Create a 5x5 grid and remove 4 corners or similar to get 21."""
    # 5x5 = 25. Remove 4.
    # Let's just generate 25 and take first 21? No, that's biased.
    # Better: 4x5 = 20, add 1 in center?
    # Or just use a dense grid of 21 points.
    # Let's try a 7x3 grid (transposed)
    return _seed_grid_3x7(W, H)

def _optimize_initial(W_init, x_init, y_init, r_init, deadline):
    """Run scipy optimization from a given initial state."""
    params0 = _pack(W_init, x_init, y_init, r_init)
    
    # Bounds
    # W in [1.0, 2.0]
    # x in [0, 2], y in [0, 2], r in [0.01, 0.5]
    bounds = [(1.0, 2.0)]
    for _ in range(N):
        bounds.append((0.0, 2.0)) # x
        bounds.append((0.0, 2.0)) # y
        bounds.append((0.01, 0.5)) # r
        
    # Method: L-BFGS-B is good for large N with bounds
    res = minimize(
        _penalty_objective, 
        params0, 
        args=(N,),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    
    if time.time() > deadline:
        return None, 0.0
        
    W, x, y, r = _unpack(res.x)
    H = 2.0 - W
    
    is_valid, val = _validate_and_compute(r, x, y, W, H)
    if is_valid:
        return np.column_stack([x, y, r]), val
    else:
        # If invalid, try to repair? Or just return 0.
        # The penalty should have driven it to valid.
        # If not, it's a local minimum in the penalty space that is still invalid.
        return None, 0.0

def construct():
    deadline = time.time() + 115.0  # Safety margin
    
    best_result = None
    best_val = 0.0
    
    # Define several initial configurations
    # 1. Square 1x1, 3x7 grid
    W1, H1 = 1.0, 1.0
    x1, y1, r1 = _seed_grid_3x7(W1, H1)
    
    # 2. Rectangle 1.2 x 0.8, 3x7 grid
    W2, H2 = 1.2, 0.8
    x2, y2, r2 = _seed_grid_3x7(W2, H2)
    
    # 3. Rectangle 1.5 x 0.5, 3x7 grid
    W3, H3 = 1.5, 0.5
    x3, y3, r3 = _seed_grid_3x7(W3, H3)
    
    # 4. Random initialization in 1x1
    W4, H4 = 1.0, 1.0
    np.random.seed(42)
    x4 = np.random.uniform(0.1, 0.9, N)
    y4 = np.random.uniform(0.1, 0.9, N)
    r4 = np.full(N, 0.05)
    
    # 5. Another random
    W5, H5 = 1.1, 0.9
    np.random.seed(123)
    x5 = np.random.uniform(0.1, 1.0, N)
    y5 = np.random.uniform(0.1, 0.8, N)
    r5 = np.full(N, 0.04)
    
    inits = [
        (W1, x1, y1, r1),
        (W2, x2, y2, r2),
        (W3, x3, y3, r3),
        (W4, x4, y4, r4),
        (W5, x5, y5, r5)
    ]
    
    for W_init, x_init, y_init, r_init in inits:
        if time.time() > deadline:
            break
            
        result, val = _optimize_initial(W_init, x_init, y_init, r_init, deadline)
        
        if result is not None and val > best_val:
            best_val = val
            best_result = result
            
        # If we have time, do a second pass with perturbed best result
        if best_result is not None and time.time() < deadline - 10:
             # Perturb best result
             x_p = best_result[:, 0].copy() + np.random.normal(0, 0.01, N)
             y_p = best_result[:, 1].copy() + np.random.normal(0, 0.01, N)
             r_p = best_result[:, 2].copy() + np.random.normal(0, 0.005, N)
             W_p = 1.0 + np.random.uniform(-0.1, 0.1)
             
             # Clamp radii
             r_p = np.clip(r_p, 0.01, 0.5)
             
             result2, val2 = _optimize_initial(W_p, x_p, y_p, r_p, deadline)
             if result2 is not None and val2 > best_val:
                 best_val = val2
                 best_result = result2

    if best_result is None:
        # Fallback: return a valid grid, even if suboptimal
        W, H = 1.0, 1.0
        x, y, r = _seed_grid_3x7(W, H)
        # Ensure validity
        # The grid should be valid.
        best_result = np.column_stack([x, y, r])
        
    return best_result

# EVOLVE-BLOCK-END
