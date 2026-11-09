import numpy as np
import time
from scipy.optimize import minimize

def _is_valid(centers, sides, tol=1e-7):
    """Check if the packing is valid."""
    n = len(sides)
    if centers.shape != (n, 2) or sides.shape != (n,):
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    
    # Check inside unit square
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
            
    # Check no overlaps
    for i in range(n):
        for j in range(i+1, n):
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            min_dx = (sides[i] + sides[j])/2 - tol
            min_dy = (sides[i] + sides[j])/2 - tol
            if dx < min_dx and dy < min_dy:
                return False
    return True

def _objective_score(centers, sides):
    """Return sum of sides if valid, else -1."""
    if not _is_valid(centers, sides):
        return -1.0
    return np.sum(sides)

def _smooth_objective(x, n=10):
    """
    Smooth objective function for optimization.
    x is a flat array of length 3*n: [cx0, cy0, s0, cx1, cy1, s1, ...]
    Returns value to be minimized: -sum(sides) + penalties.
    """
    centers = x[:2*n].reshape(n, 2)
    sides = x[2*n:]
    
    # Ensure sides are positive for calculation stability, but penalize if small
    # We want to maximize sides, so we minimize -sum(sides)
    sum_sides = np.sum(sides)
    
    penalty = 0.0
    
    # Boundary penalties
    # cx - s/2 >= 0  => s/2 - cx <= 0
    # cx + s/2 <= 1  => cx + s/2 - 1 <= 0
    # Use squared violations
    v_left = np.maximum(0, sides/2 - centers[:, 0])
    v_right = np.maximum(0, centers[:, 0] + sides/2 - 1)
    v_bottom = np.maximum(0, sides/2 - centers[:, 1])
    v_top = np.maximum(0, centers[:, 1] + sides/2 - 1)
    
    boundary_pen = np.sum(v_left**2) + np.sum(v_right**2) + np.sum(v_bottom**2) + np.sum(v_top**2)
    penalty += 100.0 * boundary_pen
    
    # Overlap penalties
    # For each pair i, j
    # dx = |cx_i - cx_j|
    # dy = |cy_i - cy_j|
    # req = (s_i + s_j)/2
    # overlap_x = max(0, req - dx)
    # overlap_y = max(0, req - dy)
    # penalty += overlap_x * overlap_y
    
    # Vectorized overlap calculation
    # centers shape (n, 2)
    # We can use broadcasting to get all pairwise differences
    # cx_diff shape (n, n)
    cx = centers[:, 0]
    cy = centers[:, 1]
    cx_diff = np.abs(cx[:, np.newaxis] - cx[np.newaxis, :])
    cy_diff = np.abs(cy[:, np.newaxis] - cy[np.newaxis, :])
    
    # sides sum shape (n, n)
    s_sum = sides[:, np.newaxis] + sides[np.newaxis, :]
    req = s_sum / 2.0
    
    # Overlap amounts
    ox = np.maximum(0, req - cx_diff)
    oy = np.maximum(0, req - cy_diff)
    
    # We only want i < j, so take upper triangle
    # Or just sum all and divide by 2, but diagonal is 0 anyway since req_ii = s_i, dx_ii=0, ox_ii = s_i/2, oy_ii = s_i/2 -> product s_i^2/4. 
    # Wait, diagonal should be 0.
    # For i=i, req = s_i, dx=0, ox = s_i/2. This is not an overlap with itself.
    # We must zero out the diagonal.
    mask = ~np.eye(n, dtype=bool)
    ox = ox * mask
    oy = oy * mask
    
    overlap_pen = np.sum(ox * oy)
    penalty += 10.0 * overlap_pen
    
    # Penalize negative sides strongly
    neg_sides = np.maximum(0, -sides)
    penalty += 1000.0 * np.sum(neg_sides**2)
    
    return -sum_sides + penalty

def _optimize_from_start(x0, max_iter=2000, tol=1e-10):
    """
    Run Nelder-Mead optimization from x0.
    Returns best valid solution found.
    """
    n = 10
    options = {
        'maxiter': max_iter,
        'xatol': tol,
        'fatol': tol,
        'adaptive': True
    }
    
    try:
        res = minimize(_smooth_objective, x0, method='Nelder-Mead', options=options)
        x_res = res.x
        centers = x_res[:2*n].reshape(n, 2)
        sides = x_res[2*n:]
        
        # Post-processing: Clip sides to positive, centers to valid range
        sides = np.maximum(sides, 1e-6)
        for i in range(n):
            s = sides[i]
            centers[i, 0] = np.clip(centers[i, 0], s/2, 1 - s/2)
            centers[i, 1] = np.clip(centers[i, 1], s/2, 1 - s/2)
            
        score = _objective_score(centers, sides)
        if score > 0:
            return centers, sides, score
    except Exception:
        pass
        
    return None, None, -1.0

def _generate_initial_configs():
    """Generate a list of diverse initial configurations."""
    configs = []
    n = 10
    
    # 1. The 3.0 packing (2 large, 8 small)
    c1 = np.zeros((n, 2))
    s1 = np.zeros(n)
    s1[0] = 0.5; c1[0] = [0.25, 0.25]
    s1[1] = 0.5; c1[1] = [0.75, 0.25]
    for i in range(2, 10):
        s1[i] = 0.25
        # 4x2 grid in top half
        row = (i - 2) // 4
        col = (i - 2) % 4
        c1[i, 0] = 0.125 + col * 0.25
        c1[i, 1] = 0.625 + row * 0.25
    configs.append(np.concatenate([c1.flatten(), s1]))
    
    # 2. Random configs
    rng = np.random.RandomState(42)
    for _ in range(50):
        # Random sides between 0.1 and 0.5
        s = rng.uniform(0.1, 0.5, n)
        # Random centers within valid range
        c = np.zeros((n, 2))
        for i in range(n):
            c[i, 0] = rng.uniform(s[i]/2, 1 - s[i]/2)
            c[i, 1] = rng.uniform(s[i]/2, 1 - s[i]/2)
        configs.append(np.concatenate([c.flatten(), s]))
        
    # 3. Grid-like configs with varying sizes
    # 3x3 grid + 1
    for scale in [0.3, 0.31, 0.32]:
        c = np.zeros((n, 2))
        s = np.zeros(n)
        for i in range(9):
            s[i] = scale
            row = i // 3
            col = i % 3
            c[i, 0] = scale/2 + col * scale
            c[i, 1] = scale/2 + row * scale
        # 10th square
        s[9] = 1 - 3*scale
        c[9, 0] = 1 - s[9]/2
        c[9, 1] = 1 - s[9]/2
        configs.append(np.concatenate([c.flatten(), s]))
        
    return configs

def squares_in_square10() -> tuple:
    """
    Returns:
        centers: np.ndarray of shape (10, 2)
        sides:   np.ndarray of shape (10,)
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    best_centers = None
    best_sides = None
    best_score = -1.0
    
    # Get initial configs
    initial_configs = _generate_initial_configs()
    
    # Sort by initial objective (smooth) to try promising ones first? 
    # Or just iterate.
    
    for i, x0 in enumerate(initial_configs):
        if time.time() > deadline:
            break
            
        # Run optimization
        # Limit iterations based on remaining time? 
        # Nelder-Mead is fast. 2000 iterations is usually < 1s.
        centers, sides, score = _optimize_from_start(x0, max_iter=2000)
        
        if score > best_score:
            best_score = score
            best_centers = centers
            best_sides = sides
            
        # If we found a very good solution, we can try to polish it
        if best_score > 3.05:
             # Run more iterations on the best found so far
             x_best = np.concatenate([best_centers.flatten(), best_sides])
             if time.time() < deadline - 10:
                 centers, sides, score = _optimize_from_start(x_best, max_iter=5000, tol=1e-12)
                 if score > best_score:
                     best_score = score
                     best_centers = centers
                     best_sides = sides

    # Fallback to known valid packing if no valid solution found
    if best_centers is None:
        c1 = np.zeros((10, 2))
        s1 = np.zeros(10)
        s1[0] = 0.5; c1[0] = [0.25, 0.25]
        s1[1] = 0.5; c1[1] = [0.75, 0.25]
        for i in range(2, 10):
            s1[i] = 0.25
            row = (i - 2) // 4
            col = (i - 2) % 4
            c1[i, 0] = 0.125 + col * 0.25
            c1[i, 1] = 0.625 + row * 0.25
        best_centers = c1
        best_sides = s1
        
    # Final validation
    if not _is_valid(best_centers, best_sides):
        # Force valid fallback
        c1 = np.zeros((10, 2))
        s1 = np.zeros(10)
        s1[0] = 0.5; c1[0] = [0.25, 0.25]
        s1[1] = 0.5; c1[1] = [0.75, 0.25]
        for i in range(2, 10):
            s1[i] = 0.25
            row = (i - 2) // 4
            col = (i - 2) % 4
            c1[i, 0] = 0.125 + col * 0.25
            c1[i, 1] = 0.625 + row * 0.25
        best_centers = c1
        best_sides = s1

    return best_centers, best_sides
