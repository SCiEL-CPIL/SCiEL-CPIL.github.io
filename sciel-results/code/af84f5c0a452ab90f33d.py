import numpy as np
from scipy.optimize import minimize
import time

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 590  # 10s margin
    
    # Variables: [x1, y1, s1, x2, y2, s2, ..., x10, y10, s10]
    # Shape: (30,)
    
    def objective(v):
        # Reshape to (10, 3)
        v = v.reshape(n, 3)
        x = v[:, 0]
        y = v[:, 1]
        s = v[:, 2]
        
        # Boundary penalty
        # Left: x - s/2 < 0  => s/2 - x > 0
        # Right: x + s/2 > 1 => x + s/2 - 1 > 0
        # Bottom: y - s/2 < 0
        # Top: y + s/2 > 1
        
        p_left = np.maximum(0, s/2 - x)
        p_right = np.maximum(0, x + s/2 - 1)
        p_bottom = np.maximum(0, s/2 - y)
        p_top = np.maximum(0, y + s/2 - 1)
        
        P_b = np.sum(p_left**2 + p_right**2 + p_bottom**2 + p_top**2)
        
        # Overlap penalty
        P_o = 0.0
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(x[i] - x[j])
                dy = abs(y[i] - y[j])
                req_x = s[i]/2 + s[j]/2
                req_y = s[i]/2 + s[j]/2
                
                ox = np.maximum(0, req_x - dx)
                oy = np.maximum(0, req_y - dy)
                P_o += ox * oy
                
        sum_s = np.sum(s)
        
        # We want to maximize sum_s, so minimize -sum_s
        # Penalty weight
        lam = 100.0
        return -sum_s + lam * (P_o + P_b)

    def is_valid(v):
        v = v.reshape(n, 3)
        x = v[:, 0]
        y = v[:, 1]
        s = v[:, 2]
        
        tol = 1e-7
        if np.any(s <= 0):
            return False
        if np.any(~np.isfinite(v)):
            return False
            
        # Boundary check
        if np.any(x - s/2 < -tol) or np.any(x + s/2 > 1 + tol):
            return False
        if np.any(y - s/2 < -tol) or np.any(y + s/2 > 1 + tol):
            return False
            
        # Overlap check
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(x[i] - x[j])
                dy = abs(y[i] - y[j])
                t = (s[i] + s[j])/2 - tol
                if dx < t and dy < t:
                    return False
        return True

    best_sum = 0.0
    best_v = None
    
    # Initial guesses
    inits = []
    
    # 1. Uniform grid 4x4 (10 squares)
    k = 4
    side = 1.0/k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    s_init = np.full(n, side)
    v_init = np.column_stack([cx, cy, s_init]).flatten()
    inits.append(v_init)
    
    # 2. 3x3 grid + 1 random
    k = 3
    side = 1.0/k
    idx = np.arange(9)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    s_init = np.full(9, side)
    # 10th square small
    cx10 = 0.5
    cy10 = 0.5
    s10 = 0.1
    cx = np.append(cx, cx10)
    cy = np.append(cy, cy10)
    s_init = np.append(s_init, s10)
    v_init = np.column_stack([cx, cy, s_init]).flatten()
    inits.append(v_init)
    
    # 3. Random
    rng = np.random.default_rng(42)
    for _ in range(5):
        x = rng.uniform(0.1, 0.9, n)
        y = rng.uniform(0.1, 0.9, n)
        s = rng.uniform(0.1, 0.4, n)
        v_init = np.column_stack([x, y, s]).flatten()
        inits.append(v_init)
        
    # 4. Structured: 1 large, 9 small
    # Large at (0.25, 0.25) side 0.5
    # 9 small side 0.25
    # This is the 2.5 sum packing.
    x = np.array([0.25] + [0.25]*9) # dummy, will overwrite
    # Let's just use the grid one which is better.
    
    for v0 in inits:
        if time.time() > deadline - 5:
            break
            
        # Bounds
        bounds = []
        for i in range(n):
            bounds.append((0, 1)) # x
            bounds.append((0, 1)) # y
            bounds.append((1e-6, 1)) # s
            
        try:
            res = minimize(objective, v0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
            
            if is_valid(res.x):
                v = res.x.reshape(n, 3)
                s = v[:, 2]
                curr_sum = np.sum(s)
                if curr_sum > best_sum:
                    best_sum = curr_sum
                    best_v = res.x
        except Exception:
            pass
            
    if best_v is None:
        # Fallback to baseline
        k = 4
        side = 1.0/k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = np.full(n, side, dtype=np.float64)
        return centers, sides
        
    v = best_v.reshape(n, 3)
    centers = v[:, :2].astype(np.float64)
    sides = v[:, 2].astype(np.float64)
    
    return centers, sides
