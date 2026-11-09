# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def squares_in_square10() -> tuple:
    """
    Maximizes the total side length of 10 axis-aligned squares in the unit square.
    Uses a penalty-based optimization with multiple restarts.
    """
    n = 10
    tol = 1e-7
    
    # Variables: [cx_1, cy_1, s_1, cx_2, cy_2, s_2, ...]
    # Total 3n variables.
    
    def objective(params):
        # params: shape (3n,)
        cx = params[0::3]
        cy = params[1::3]
        s = params[2::3]
        
        # Sum of sides to maximize (so we minimize negative)
        obj = -np.sum(s)
        
        # Boundary penalties
        # Squares must be in [0, 1]
        # cx - s/2 >= 0 => s/2 - cx <= 0
        # cx + s/2 <= 1 => cx + s/2 - 1 <= 0
        # Same for y
        
        b_pen = 0.0
        for i in range(n):
            # Bottom
            b_pen += max(0.0, s[i]/2.0 - cy[i])
            # Top
            b_pen += max(0.0, cy[i] + s[i]/2.0 - 1.0)
            # Left
            b_pen += max(0.0, s[i]/2.0 - cx[i])
            # Right
            b_pen += max(0.0, cx[i] + s[i]/2.0 - 1.0)
            
        # Overlap penalties
        o_pen = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                # Distance between centers
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                
                # Required distance to not overlap
                req = (s[i] + s[j]) / 2.0
                
                # Overlap depth in x
                ox = max(0.0, req - dx)
                # Overlap depth in y
                oy = max(0.0, req - dy)
                
                # If both > 0, they overlap. Penalty is product.
                o_pen += ox * oy
                
        # Weights
        w_b = 1000.0
        w_o = 1000.0
        
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
                
                # Overlap condition: |cx_i - cx_j| < (s_i + s_j)/2 - tol AND |cy_i - cy_j| < (s_i + s_j)/2 - tol
                # Wait, the task says: overlap if |cx_i - cx_j| < t and |cy_i - cy_j| < t, where t = (s_i + s_j)/2 - tol
                # This means if |cx_i - cx_j| >= (s_i + s_j)/2 - tol, they are NOT considered overlapping (touching is allowed).
                # So valid if NOT (|cx_i - cx_j| < (s_i + s_j)/2 - tol AND |cy_i - cy_j| < (s_i + s_j)/2 - tol)
                
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
    
    # Time limit
    start_time = time.time()
    time_limit = 590.0 # Safety margin
    
    # Initial guesses
    initial_guesses = []
    
    # 1. 4x4 Grid
    k = 4
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx_grid = (cols + 0.5) * side
    cy_grid = (rows + 0.5) * side
    params_grid = np.zeros(3 * n)
    params_grid[0::3] = cx_grid
    params_grid[1::3] = cy_grid
    params_grid[2::3] = side
    initial_guesses.append(params_grid)
    
    # 2. Random restarts
    np.random.seed(42)
    for _ in range(50):
        cx_r = np.random.uniform(0.1, 0.9, n)
        cy_r = np.random.uniform(0.1, 0.9, n)
        s_r = np.random.uniform(0.1, 0.4, n)
        params_r = np.zeros(3 * n)
        params_r[0::3] = cx_r
        params_r[1::3] = cy_r
        params_r[2::3] = s_r
        initial_guesses.append(params_r)
        
    # 3. Heuristic: 4 large corners, 1 center, 5 small
    # This is a bit complex to set up perfectly, so we'll rely on random + grid.
    
    # Optimization
    for guess in initial_guesses:
        if time.time() - start_time > time_limit:
            break
            
        try:
            res = minimize(objective, guess, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-10})
            
            params = res.x
            cx = params[0::3]
            cy = params[1::3]
            s = params[2::3]
            
            centers = np.column_stack((cx, cy))
            
            # Check validity
            if is_valid(centers, s):
                score = get_score(centers, s)
                if score > best_score:
                    best_score = score
                    best_centers = centers.copy()
                    best_sides = s.copy()
            else:
                # Try to fix by shrinking squares slightly
                # This might reduce score but ensure validity
                # We can scale sides down until valid
                # Simple approach: reduce all sides by a small factor
                # But this might not fix local overlaps.
                # For now, skip invalid results. The penalty method should handle most.
                pass
                
        except Exception as e:
            pass

    # If no valid solution found (unlikely), return baseline
    if best_centers is None:
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

# EVOLVE-BLOCK-END
