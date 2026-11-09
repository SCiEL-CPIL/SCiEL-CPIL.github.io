# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def squares_in_square10() -> tuple:
    """
    Pack 10 axis-aligned squares in [0,1]^2 maximizing sum of side lengths.
    Uses a structured packing: 3 large squares in top row, 5 medium in middle,
    2 large in bottom, then local refinement.
    """
    t0 = time.time()
    deadline = t0 + 580.0  # safety margin before 630s kill

    n = 10
    tol = 1e-7

    def is_valid(centers, sides):
        if centers.shape != (n, 2) or sides.shape != (n,):
            return False
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return False
        if np.any(sides <= 0):
            return False
        # boundary
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if cx - s/2 < -tol or cx + s/2 > 1 + tol:
                return False
            if cy - s/2 < -tol or cy + s/2 > 1 + tol:
                return False
        # overlap
        for i in range(n):
            for j in range(i+1, n):
                t = (sides[i] + sides[j]) / 2 - tol
                if abs(centers[i,0] - centers[j,0]) < t and abs(centers[i,1] - centers[j,1]) < t:
                    return False
        return True

    def objective(x):
        """x: [cx0,cy0,s0, cx1,cy1,s1, ...] -> negative sum of sides + penalty for violations"""
        centers = x[0::3].reshape(-1, 2)[:, :2] if x.size == 3*n else None
        # Actually x is flat: for each i: cx, cy, s
        cx = x[0::3]
        cy = x[1::3]
        s = x[2::3]
        centers = np.column_stack([cx, cy])
        
        # Penalty for going outside [0,1]
        pen = 0.0
        for i in range(n):
            if cx[i] - s[i]/2 < 0: pen += (-(cx[i] - s[i]/2))**2 * 1e6
            if cx[i] + s[i]/2 > 1: pen += ((cx[i] + s[i]/2) - 1)**2 * 1e6
            if cy[i] - s[i]/2 < 0: pen += (-(cy[i] - s[i]/2))**2 * 1e6
            if cy[i] + s[i]/2 > 1: pen += ((cy[i] + s[i]/2) - 1)**2 * 1e6
        
        # Penalty for overlap
        for i in range(n):
            for j in range(i+1, n):
                t = (s[i] + s[j]) / 2
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                # If both |dx| < t and |dy| < t, overlap
                if abs(dx) < t and abs(dy) < t:
                    # penalty proportional to how deep the overlap is
                    pen += (t - abs(dx))**2 * 1e4 + (t - abs(dy))**2 * 1e4
        
        return -np.sum(s) + pen

    # Structured initial guess:
    # Top row: 3 squares of side 1/3
    # Middle row: 5 squares
    # Bottom: 2 squares
    #
    # Let's design a good packing:
    # Top: 3 squares, side = 1/3, y_center = 1 - 1/6 = 5/6
    #   x_centers: 1/6, 1/2, 5/6
    #
    # Remaining area below y=1/3: height 1/3
    # We can fit 2 squares of side 1/3 in bottom? No, that's same as top.
    #
    # Better: 
    # Top row (y from 1/3 to 1): 3 squares of side 1/3
    #   Centers: (1/6, 5/6), (1/2, 5/6), (5/6, 5/6), sides = 1/3
    #
    # Bottom area (y from 0 to 1/3): height 1/3, width 1
    #   Place 2 squares of side 1/3? That uses width 2/3, leaving 1/3.
    #   Or place squares of different sizes.
    #
    # Let me try a known good packing for 10 squares:
    # From literature, optimal sum for 10 squares in unit square is about 3.5+
    #
    # Design:
    # - 3 squares of side a=0.32 in top
    # - 4 squares of side b=0.25 in middle  
    # - 3 squares of side c=0.20 in bottom
    #
    # Let's just use a reasonable starting point and let optimizer work.
    
    # Starting configuration:
    # 3 large on top: side ~0.33
    # 4 medium in middle: side ~0.25
    # 3 small on bottom: side ~0.20
    
    cx_init = np.array([
        1/6, 1/2, 5/6,   # top row
        1/8, 3/8, 5/8, 7/8,  # middle row
        1/6, 1/2, 5/6    # bottom row
    ])
    cy_init = np.array([
        5/6, 5/6, 5/6,   # top
        0.42, 0.42, 0.42, 0.42,  # middle
        0.15, 0.15, 0.15  # bottom
    ])
    s_init = np.array([
        0.33, 0.33, 0.33,
        0.25, 0.25, 0.25, 0.25,
        0.20, 0.20, 0.20
    ])
    
    x0 = np.concatenate([cx_init, cy_init, s_init])
    
    # Bounds: centers in [0,1], sides in [0.01, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # cx
        bounds.append((0.0, 1.0))  # cy
        bounds.append((0.01, 0.5))  # s
    
    best_x = x0.copy()
    best_val = np.sum(s_init)
    
    # Try multiple optimization runs with different strategies
    for attempt in range(8):
        if time.time() > deadline - 60:
            break
        
        if attempt == 0:
            x_start = x0
        elif attempt == 1:
            # Slightly perturbed
            x_start = x0 + np.random.normal(0, 0.02, x0.size)
        elif attempt == 2:
            # Different initial: more uniform
            cx_i = np.array([0.2, 0.5, 0.8, 0.15, 0.35, 0.55, 0.75, 0.25, 0.5, 0.75])
            cy_i = np.array([0.75, 0.75, 0.75, 0.5, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25])
            s_i = np.array([0.3, 0.3, 0.3, 0.22, 0.22, 0.22, 0.22, 0.25, 0.25, 0.25])
            x_start = np.concatenate([cx_i, cy_i, s_i])
        elif attempt == 3:
            # Try: 2 big, 4 medium, 4 small
            cx_i = np.array([0.25, 0.75, 0.15, 0.35, 0.65, 0.85, 0.1, 0.5, 0.9, 0.3])
            cy_i = np.array([0.7, 0.7, 0.45, 0.45, 0.45, 0.45, 0.2, 0.2, 0.2, 0.2])
            s_i = np.array([0.35, 0.35, 0.2, 0.2, 0.2, 0.2, 0.25, 0.25, 0.25, 0.25])
            x_start = np.concatenate([cx_i, cy_i, s_i])
        else:
            # Random perturbation of best found
            x_start = best_x + np.random.normal(0, 0.03, best_x.size)
        
        # Clip to bounds
        for i in range(n):
            x_start[3*i] = np.clip(x_start[3*i], 0.01, 0.99)
            x_start[3*i+1] = np.clip(x_start[3*i+1], 0.01, 0.99)
            x_start[3*i+2] = np.clip(x_start[3*i+2], 0.02, 0.49)
        
        try:
            result = minimize(
                objective, x_start, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.success or result.fun < objective(x_start):
                cx = result.x[0::3]
                cy = result.x[1::3]
                s = result.x[2::3]
                centers = np.column_stack([cx, cy])
                if is_valid(centers, s):
                    val = np.sum(s)
                    if val > best_val:
                        best_val = val
                        best_x = result.x.copy()
        except Exception:
            pass
    
    # Final local refinement on best
    if time.time() < deadline - 30:
        try:
            result = minimize(
                objective, best_x, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            cx = result.x[0::3]
            cy = result.x[1::3]
            s = result.x[2::3]
            centers = np.column_stack([cx, cy])
            if is_valid(centers, s):
                val = np.sum(s)
                if val > best_val:
                    best_val = val
                    best_x = result.x.copy()
        except Exception:
            pass
    
    cx = best_x[0::3]
    cy = best_x[1::3]
    s = best_x[2::3]
    centers = np.column_stack([cx, cy]).astype(np.float64)
    sides = s.astype(np.float64)
    
    # Final validity check; if invalid, fall back to grid
    if not is_valid(centers, sides):
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


# EVOLVE-BLOCK-END
