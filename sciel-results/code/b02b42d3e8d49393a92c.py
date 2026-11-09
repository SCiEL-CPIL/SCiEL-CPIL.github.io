import numpy as np
from scipy.optimize import minimize
import time

def squares_in_square10() -> tuple:
    n = 10
    tol = 1e-7
    
    # Variables: x_1, y_1, s_1, ..., x_n, y_n, s_n
    # Total 3n variables.
    
    def objective(params):
        x = params[0::3]
        y = params[1::3]
        s = params[2::3]
        
        # Maximize sum of sides -> Minimize -sum(s)
        obj = -np.sum(s)
        
        # Boundary Penalties
        # x - s/2 >= 0, x + s/2 <= 1
        # y - s/2 >= 0, y + s/2 <= 1
        b_pen = 0.0
        for i in range(n):
            b_pen += max(0.0, s[i]/2.0 - x[i])**2
            b_pen += max(0.0, x[i] + s[i]/2.0 - 1.0)**2
            b_pen += max(0.0, s[i]/2.0 - y[i])**2
            b_pen += max(0.0, y[i] + s[i]/2.0 - 1.0)**2
            
        # Overlap Penalties
        o_pen = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = abs(x[i] - x[j])
                dy = abs(y[i] - y[j])
                req_x = (s[i] + s[j]) / 2.0
                req_y = (s[i] + s[j]) / 2.0
                
                # Overlap depth
                ox = max(0.0, req_x - dx)
                oy = max(0.0, req_y - dy)
                
                # Penalty is area of overlap squared to be smooth and strong
                o_pen += (ox * oy)**2
                
        # Weights
        w_b = 1000.0
        w_o = 10000.0 # Stronger to ensure non-overlap
        
        return obj + w_b * b_pen + w_o * o_pen

    def is_valid(centers, sides):
        # Check boundaries
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if cx - s/2.0 < -tol or cx + s/2.0 > 1.0 + tol:
                return False
            if cy - s/2.0 < -tol or cy + s/2.0 > 1.0 + tol:
                return False
            if s <= 0:
                return False
            if not np.isfinite(cx) or not np.isfinite(cy) or not np.isfinite(s):
                return False
        
        # Check overlaps
        for i in range(n):
            for j in range(i + 1, n):
                cx_i, cy_i = centers[i]
                cx_j, cy_j = centers[j]
                s_i = sides[i]
                s_j = sides[j]
                
                dx = abs(cx_i - cx_j)
                dy = abs(cy_i - cy_j)
                t = (s_i + s_j) / 2.0 - tol
                
                if dx < t and dy < t:
                    return False
                    
        return True

    def get_score(centers, sides):
        return np.sum(sides)

    best_score = 0.0
    best_centers = None
    best_sides = None
    
    start_time = time.time()
    time_limit = 590.0
    
    initial_guesses = []
    
    # 1. Grid 3x3 + 1 center (shrunken)
    # 9 squares of 1/3, 1 square in middle
    # This is a good starting point to escape the 1.0 seed (which was 2.5)
    # We want to reach ~3.0+
    
    # Setup: 3x3 grid of squares with side 0.32 (slightly less than 1/3 to allow gaps?)
    # Actually, let's start with 1/3 and let optimizer shrink.
    s0 = 1.0 / 3.0
    params = np.zeros(3 * n)
    idx = 0
    for r in range(3):
        for c in range(3):
            if idx < n:
                cx = (c + 0.5) * s0
                cy = (r + 0.5) * s0
                params[idx*3] = cx
                params[idx*3+1] = cy
                params[idx*3+2] = s0
                idx += 1
    # 10th square in center
    if idx < n:
        params[idx*3] = 0.5
        params[idx*3+1] = 0.5
        params[idx*3+2] = s0 * 0.5 # Smaller to fit?
        idx += 1
    initial_guesses.append(params)
    
    # 2. Random restarts with larger initial sizes
    np.random.seed(42)
    for _ in range(100):
        x_r = np.random.uniform(0.2, 0.8, n)
        y_r = np.random.uniform(0.2, 0.8, n)
        s_r = np.random.uniform(0.2, 0.4, n)
        params_r = np.zeros(3 * n)
        params_r[0::3] = x_r
        params_r[1::3] = y_r
        params_r[2::3] = s_r
        initial_guesses.append(params_r)
        
    # 3. Specific heuristic: 4 corners large, others smaller
    # Corners: (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75) with s=0.4
    # This might not fit well, but gives diversity.
    params_h = np.zeros(3 * n)
    corners = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
    for i, (cx, cy) in enumerate(corners):
        params_h[i*3] = cx
        params_h[i*3+1] = cy
        params_h[i*3+2] = 0.4
    # Remaining 6 in middle
    for i in range(4, 10):
        params_h[i*3] = 0.5
        params_h[i*3+1] = 0.5
        params_h[i*3+2] = 0.2
    initial_guesses.append(params_h)

    for guess in initial_guesses:
        if time.time() - start_time > time_limit:
            break
            
        try:
            res = minimize(objective, guess, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12})
            
            params = res.x
            x = params[0::3]
            y = params[1::3]
            s = params[2::3]
            
            centers = np.column_stack((x, y))
            
            if is_valid(centers, s):
                score = get_score(centers, s)
                if score > best_score:
                    best_score = score
                    best_centers = centers.copy()
                    best_sides = s.copy()
            else:
                # Try to repair: shrink squares slightly if they are very close to boundary or overlapping
                # If invalid, skip. The penalty should handle it.
                pass
                
        except Exception as e:
            pass

    # If no valid solution found
    if best_centers is None:
        # Fallback to seed-like solution
        k = 4
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx_grid = (cols + 0.5) * side
        cy_grid = (rows + 0.5) * side
        best_centers = np.column_stack((cx_grid, cy_grid))
        best_sides = np.full(n, side)
        
    return best_centers, best_sides
