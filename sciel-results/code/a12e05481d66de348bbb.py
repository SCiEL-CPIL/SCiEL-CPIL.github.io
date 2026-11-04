"""Heilbronn triangles: 11 points in unit equilateral triangle, maximize min triangle area."""
import numpy as np
from scipy.optimize import minimize
import math

N = 11
H = math.sqrt(3) / 2

def _min_area_vec(pts):
    """
    Vectorized calculation of minimum triangle area.
    pts: (N, 2) array
    """
    min_a = float('inf')
    for i in range(N-2):
        pi = pts[i]
        for j in range(i+1, N-1):
            pj = pts[j]
            pk = pts[j+1:]
            v1 = pj - pi
            v2 = pk - pi
            cross = v1[0]*v2[1] - v1[1]*v2[0]
            areas = 0.5 * np.abs(cross)
            local_min = np.min(areas)
            if local_min < min_a:
                min_a = local_min
    return min_a

def _eval_symmetric(vars_flat):
    """
    vars_flat: array of 11 elements
    [y_center, x1, y1, x2, y2, ..., x5, y5]
    x_i are in [0, 0.5]
    y_i are in [0, H]
    """
    y_c = vars_flat[0]
    # Clamp y_c
    if y_c < 0: y_c = 0
    if y_c > H: y_c = H
    
    pts = [[0.5, y_c]]
    
    for i in range(5):
        x = vars_flat[1 + 2*i]
        y = vars_flat[2 + 2*i]
        
        # Clamp x to [0, 0.5]
        if x < 0: x = 0
        if x > 0.5: x = 0.5
        
        # Clamp y to [0, H]
        if y < 0: y = 0
        if y > H: y = H
        
        # Check if point (x, y) is inside the triangle
        # Left edge: y <= sqrt(3) * x  => x >= y/sqrt(3)
        # Right edge: y <= sqrt(3) * (1-x) => x <= 1 - y/sqrt(3)
        # Since x <= 0.5, the right edge constraint is x <= 1 - y/sqrt(3)
        # The left edge constraint is x >= y/sqrt(3)
        
        min_x = y / math.sqrt(3)
        max_x = 1 - y / math.sqrt(3)
        
        if max_x < 0.5:
             max_x = 0.5 # This shouldn't happen if y is small enough, but just in case
        if min_x > 0.5:
             # Point cannot exist at this y with x <= 0.5?
             # If y is high, the triangle narrows.
             # If y > sqrt(3)/2 * 0.5 = sqrt(3)/4, then min_x > 0.5?
             # No, min_x = y/sqrt(3). If y=H, min_x = 0.5.
             # So x must be 0.5.
             x = 0.5
        else:
             if x < min_x: x = min_x
             if x > max_x: x = max_x
             if x > 0.5: x = 0.5
             
        pts.append([x, y])
        pts.append([1-x, y])
        
    pts = np.array(pts)
    return -_min_area_vec(pts)

def construct():
    # Initial guess for symmetric parameters
    # y_center ~ H/3
    # Pairs: spread out
    init_vars = np.array([
        H/3, 
        0.25, H/4, 
        0.1, H/2, 
        0.4, H/4, 
        0.2, H/6, 
        0.3, H/6
    ])
    
    best_val = float('inf')
    best_vars = init_vars
    
    # Run multiple random restarts on the symmetric space
    rng = np.random.RandomState(42)
    
    for i in range(20):
        # Random initialization in valid ranges
        y_c = rng.random() * H
        vars_flat = np.array([y_c])
        for j in range(5):
            x = rng.random() * 0.5
            y = rng.random() * H
            vars_flat = np.append(vars_flat, [x, y])
            
        if i == 0:
            vars_flat = init_vars
            
        try:
            res = minimize(_eval_symmetric, vars_flat, method='Nelder-Mead',
                           options={'maxiter': 20000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_vars = res.x
        except Exception:
            pass
            
    # Final polish
    try:
        res_final = minimize(_eval_symmetric, best_vars, method='Nelder-Mead',
                             options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
        if res_final.fun < best_val:
            best_val = res_final.fun
            best_vars = res_final.x
    except Exception:
        pass

    # Reconstruct points
    y_c = best_vars[0]
    if y_c < 0: y_c = 0
    if y_c > H: y_c = H
    
    pts = [[0.5, y_c]]
    for i in range(5):
        x = best_vars[1 + 2*i]
        y = best_vars[2 + 2*i]
        if x < 0: x = 0
        if x > 0.5: x = 0.5
        if y < 0: y = 0
        if y > H: y = H
        
        min_x = y / math.sqrt(3)
        max_x = 1 - y / math.sqrt(3)
        if x < min_x: x = min_x
        if x > max_x: x = max_x
        if x > 0.5: x = 0.5
        
        pts.append([x, y])
        pts.append([1-x, y])
        
    return np.array(pts)
