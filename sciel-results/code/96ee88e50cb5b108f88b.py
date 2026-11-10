import numpy as np
from scipy.optimize import minimize
import time

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin
    
    best_score = -np.inf
    best_centers = None
    best_sides = None
    
    # Objective function to minimize: -sum(sides) + penalties
    def objective(x):
        # x: [cx0, cy0, s0, cx1, cy1, s1, ...]
        cx = x[0::3]
        cy = x[1::3]
        s = x[2::3]
        
        # Side lengths must be positive
        # If s <= 0, it's invalid. We can penalize this.
        # But Nelder-Mead doesn't handle bounds well natively without transformation.
        # We'll just add a penalty for s <= 0.
        pen_s = np.sum(np.maximum(0, -s)) * 1000
        
        # Boundary penalty
        # Square i is in [cx_i - s_i/2, cx_i + s_i/2]
        # Must be in [0, 1]
        # Violation if cx_i - s_i/2 < 0 or cx_i + s_i/2 > 1
        pen_bound = 0.0
        pen_bound += np.sum(np.maximum(0, s/2 - cx))
        pen_bound += np.sum(np.maximum(0, cx + s/2 - 1))
        pen_bound += np.sum(np.maximum(0, s/2 - cy))
        pen_bound += np.sum(np.maximum(0, cy + s/2 - 1))
        pen_bound *= 1000
        
        # Overlap penalty
        # Vectorized pairwise differences
        cx_diff = np.abs(cx[:, None] - cx[None, :])
        cy_diff = np.abs(cy[:, None] - cy[None, :])
        s_sum_half = (s[:, None] + s[None, :]) / 2.0
        
        # Overlap in x: s_sum_half - cx_diff (if positive)
        # Overlap in y: s_sum_half - cy_diff (if positive)
        ov_x = np.maximum(0, s_sum_half - cx_diff)
        ov_y = np.maximum(0, s_sum_half - cy_diff)
        
        # Intersection area
        inter_area = ov_x * ov_y
        # Exclude diagonal
        inter_area[np.diag_indices(n)] = 0
        pen_overlap = np.sum(inter_area) * 1000
        
        # We want to maximize sum(s), so minimize -sum(s) + penalty
        return -np.sum(s) + pen_s + pen_bound + pen_overlap

    def run_optimization(x0, max_iter=2000):
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            return res.x
        except Exception:
            return x0

    def is_valid(centers, sides, tol=1e-7):
        # Check bounds
        if np.any(centers[:, 0] - sides/2 < -tol) or np.any(centers[:, 0] + sides/2 > 1 + tol) or \
           np.any(centers[:, 1] - sides/2 < -tol) or np.any(centers[:, 1] + sides/2 > 1 + tol):
            return False
        # Check sides > 0
        if np.any(sides <= 0):
            return False
        # Check overlaps
        for i in range(n):
            for j in range(i+1, n):
                t = (sides[i] + sides[j])/2 - tol
                if abs(centers[i,0] - centers[j,0]) < t and abs(centers[i,1] - centers[j,1]) < t:
                    return False
        return True

    # Generate starting points
    starts = []
    
    # 1. 4x4 grid (16 cells, pick 10)
    # Actually, let's just use a 4x4 grid of 16 squares and keep the first 10?
    # Or better, distribute 10 squares in a 4x4 grid.
    k = 4
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx0 = (cols + 0.5) * side
    cy0 = (rows + 0.5) * side
    s0 = np.full(n, side)
    x0 = np.empty(n*3)
    x0[0::3] = cx0
    x0[1::3] = cy0
    x0[2::3] = s0
    starts.append(x0)
    
    # 2. 3x3 grid + 1 small
    # 9 squares of 1/3, 1 square of 0.1?
    # Let's try to fit 10 squares in a 3x4 grid (12 cells).
    kx, ky = 4, 3
    side_x = 1.0 / kx
    side_y = 1.0 / ky
    # This is not square. We need squares.
    # Let's stick to square grids.
    
    # 3. Random starts
    np.random.seed(42)
    for i in range(20):
        cx = np.random.uniform(0.2, 0.8, n)
        cy = np.random.uniform(0.2, 0.8, n)
        s = np.random.uniform(0.2, 0.35, n)
        x = np.empty(n*3)
        x[0::3] = cx
        x[1::3] = cy
        x[2::3] = s
        starts.append(x)
        
    # 4. "Strip" packing: 2 rows of 5
    # 5 squares of 0.2 in row 1, 5 squares of 0.2 in row 2?
    # Side 0.2. Sum 2.0.
    # Let's try larger.
    # 5 squares of 0.3 in row 1? Width 1.5 > 1. No.
    # 3 squares of 0.33 in row 1? Width 1.0.
    # 3 squares of 0.33 in row 2?
    # 3 squares of 0.33 in row 3?
    # 1 square of 0.33 in row 4?
    # Total 10.
    # Let's construct this.
    s_strip = 1.0/3
    cx_strip = []
    cy_strip = []
    s_strip_list = []
    # Row 0: 3 squares
    for i in range(3):
        cx_strip.append((i + 0.5) * s_strip)
        cy_strip.append(0.5 * s_strip)
        s_strip_list.append(s_strip)
    # Row 1: 3 squares
    for i in range(3):
        cx_strip.append((i + 0.5) * s_strip)
        cy_strip.append(1.5 * s_strip)
        s_strip_list.append(s_strip)
    # Row 2: 3 squares
    for i in range(3):
        cx_strip.append((i + 0.5) * s_strip)
        cy_strip.append(2.5 * s_strip)
        s_strip_list.append(s_strip)
    # Row 3: 1 square (centered)
    cx_strip.append(0.5)
    cy_strip.append(0.5 * s_strip) # This overlaps with row 0?
    # 3 rows of 3 squares take height 3 * (1/3) = 1.0. No space for row 3.
    # So 3x3 grid is full.
    # To add a 10th square, we must shrink the 9.
    # Let's try 4x3 grid of squares?
    # Side 1/4 = 0.25.
    # 4 cols, 3 rows = 12 squares.
    # We use 10.
    cx_43 = []
    cy_43 = []
    s_43 = []
    s_43_val = 0.25
    for r in range(3):
        for c in range(4):
            if len(cx_43) < 10:
                cx_43.append((c + 0.5) * s_43_val)
                cy_43.append((r + 0.5) * s_43_val)
                s_43.append(s_43_val)
    x_43 = np.empty(n*3)
    x_43[0::3] = cx_43
    x_43[1::3] = cy_43
    x_43[2::3] = s_43
    starts.append(x_43)
    
    # 5. "L" shape: 2 large, 8 small
    # 2 large squares of 0.5 in corners?
    # (0.25, 0.25) and (0.75, 0.75)?
    # This leaves a cross shape.
    # Let's just add this as a start.
    cx_l = [0.25, 0.75, 0.5, 0.25, 0.75, 0.5, 0.25, 0.75, 0.5, 0.5]
    cy_l = [0.25, 0.75, 0.25, 0.5, 0.5, 0.75, 0.75, 0.25, 0.5, 0.5]
    s_l = [0.5, 0.5, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    x_l = np.empty(n*3)
    x_l[0::3] = cx_l
    x_l[1::3] = cy_l
    x_l[2::3] = s_l
    starts.append(x_l)

    # Optimize each start
    for x0 in starts:
        if time.time() > deadline - 10:
            break
            
        x_res = run_optimization(x0)
        
        # Extract results
        cx = x_res[0::3]
        cy = x_res[1::3]
        s = x_res[2::3]
        
        # Validate
        if np.any(s <= 0) or np.any(~np.isfinite(cx)) or np.any(~np.isfinite(cy)) or np.any(~np.isfinite(s)):
            continue
            
        centers = np.column_stack((cx, cy))
        sides = s
        
        if is_valid(centers, sides):
            score = np.sum(sides)
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
                best_sides = sides.copy()
                
    # If no valid solution found, return baseline (which is valid)
    if best_centers is None:
        k = 4
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = np.full(n, side, dtype=np.float64)
        return centers, sides

    return best_centers, best_sides
