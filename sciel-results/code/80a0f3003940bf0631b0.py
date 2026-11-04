"""Heilbronn triangles: 11 points in unit equilateral triangle, maximize min triangle area."""
import numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import combinations
import math

N = 11
H = math.sqrt(3) / 2

def _tri_area(p1, p2, p3):
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_area(pts):
    min_a = float('inf')
    for i, j, k in combinations(range(len(pts)), 3):
        a = _tri_area(pts[i], pts[j], pts[k])
        if a < min_a:
            min_a = a
    return min_a

def _inside(p, m=1e-9):
    x, y = p
    if y < -m or x < -m or x > 1 + m:
        return False
    if y > math.sqrt(3) * x + m:
        return False
    if y > math.sqrt(3) * (1 - x) + m:
        return False
    return True

def _project(p, m=1e-8):
    x, y = p
    if y < m:
        y = m
    min_x = y / math.sqrt(3)
    if x < min_x:
        x = min_x
    max_x = 1 - y / math.sqrt(3)
    if x > max_x:
        x = max_x
    if x < m:
        x = m
    if x > 1 - m:
        x = 1 - m
    return [x, y]

def construct():
    # Strategy: Exploit symmetry for N=11.
    # Assume the 3 vertices are included.
    # The remaining 8 points form 4 symmetric pairs about x=0.5.
    # Variables: x1, y1, x2, y2, x3, y3, x4, y4 where xi in [0, 0.5].
    
    def _build_points(params):
        x1, y1, x2, y2, x3, y3, x4, y4 = params
        # Ensure points are inside the triangle
        # For a point (x, y) with x in [0, 0.5], the constraint is y <= sqrt(3)*x
        # Also y >= 0.
        # The symmetric point is (1-x, y). For x in [0, 0.5], 1-x is in [0.5, 1].
        # Constraint for (1-x, y) is y <= sqrt(3)*(1-(1-x)) = sqrt(3)*x. Same constraint.
        
        pts = [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, H],
            [x1, y1], [1.0 - x1, y1],
            [x2, y2], [1.0 - x2, y2],
            [x3, y3], [1.0 - x3, y3],
            [x4, y4], [1.0 - x4, y4]
        ]
        
        # Project points to ensure they are inside
        # Note: x1 is in [0, 0.5]. If y1 > sqrt(3)*x1, we must clamp.
        # But the optimizer will handle bounds. We just need to define valid bounds.
        # For x in [0, 0.5], y must be in [0, sqrt(3)*x].
        # This is a non-rectangular bound. We can use a penalty or transformation.
        # Transformation: Let y = x * t * sqrt(3) where t in [0, 1].
        # Then y <= sqrt(3)*x is satisfied.
        # So variables: x1, t1, x2, t2, x3, t3, x4, t4.
        # x_i in [0, 0.5], t_i in [0, 1].
        
        pts = [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, H],
            [x1, x1 * t1 * math.sqrt(3)], [1.0 - x1, x1 * t1 * math.sqrt(3)],
            [x2, x2 * t2 * math.sqrt(3)], [1.0 - x2, x2 * t2 * math.sqrt(3)],
            [x3, x3 * t3 * math.sqrt(3)], [1.0 - x3, x3 * t3 * math.sqrt(3)],
            [x4, x4 * t4 * math.sqrt(3)], [1.0 - x4, x4 * t4 * math.sqrt(3)]
        ]
        
        # Final safety projection
        pts = np.array([_project(p) for p in pts])
        return pts

    def _neg_min_area_sym(params):
        pts = _build_points(params)
        return -_min_area(pts)

    # Bounds for x_i: [0, 0.5], t_i: [0, 1]
    bounds = [(0, 0.5), (0, 1)] * 4
    
    # Initial guess
    # A reasonable guess: spread points out.
    # x: 0.1, 0.2, 0.3, 0.4
    # t: 0.5, 0.5, 0.5, 0.5
    x0 = [0.1, 0.5, 0.2, 0.5, 0.3, 0.5, 0.4, 0.5]
    
    # Run Differential Evolution
    res_de = differential_evolution(
        _neg_min_area_sym, 
        bounds, 
        seed=42, 
        maxiter=200, 
        popsize=15,
        tol=1e-6,
        init='sobol',
        polish=True
    )
    
    best_params = res_de.x
    best_val = -res_de.fun
    
    # Polish with Nelder-Mead on the transformed variables
    res_nm = minimize(_neg_min_area_sym, best_params, method='Nelder-Mead',
                      options={'maxiter': 20000, 'xatol': 1e-12, 'fatol': 1e-14})
    
    if res_nm.fun < res_de.fun:
        best_params = res_nm.x
        best_val = -res_nm.fun
    
    points = _build_points(best_params)
    
    # Optional: Try a full 22-variable polish from this symmetric start to break symmetry if needed
    # This ensures we don't miss a non-symmetric local optimum that is better.
    flat = points.flatten()
    
    def _neg_min_area_full(flat):
        pts = flat.reshape(N, 2)
        pts = np.array([_project(p) for p in pts])
        return -_min_area(pts)
        
    res_full = minimize(_neg_min_area_full, flat, method='Nelder-Mead',
                        options={'maxiter': 20000, 'xatol': 1e-12, 'fatol': 1e-14})
    
    if res_full.fun < -best_val:
        points = res_full.x.reshape(N, 2)
        points = np.array([_project(p) for p in points])
        
    return points
