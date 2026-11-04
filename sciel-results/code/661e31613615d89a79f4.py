"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time
import random

N = 13

# EVOLVE-BLOCK-START
def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 points."""
    n = len(points)
    min_area = float('inf')
    # Vectorized approach for speed
    # i, j, k indices
    # We can generate all combinations
    # But for N=13, a simple loop is sufficient and less memory overhead
    for i in range(n):
        p1 = points[i]
        for j in range(i + 1, n):
            p2 = points[j]
            v1 = p2 - p1
            for k in range(j + 1, n):
                p3 = points[k]
                v2 = p3 - p1
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute the area of the convex hull of the given points."""
    # For points on a circle, this is fixed.
    # For general points, we use a simple O(N log N) or O(N^2) hull.
    # Since N=13, O(N^2) is fine.
    # Sort by x, then y
    pts = points.copy()
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    # Build lower hull
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
        
    # Build upper hull
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
        
    # Concatenation of lower and upper hull gives the convex hull
    # Pop the last point of each half because it's repeated by the other
    hull = lower[:-1] + upper[:-1]
    
    if len(hull) < 3:
        return 0.0
        
    # Shoelace formula
    area = 0.0
    m = len(hull)
    for i in range(m):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % m]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0

def _score(points):
    """Calculate the score: min triangle area / convex hull area."""
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area == 0:
        return 0.0
    return min_area / hull_area

def _soft_min_objective(theta_vars, beta):
    """
    Objective function for optimization.
    Minimizes the negative soft-min of triangle areas.
    theta_vars: array of angles for points 1 to N-1. Point 0 is fixed at 0.
    """
    thetas = np.zeros(N)
    thetas[0] = 0.0
    thetas[1:] = theta_vars
    
    # Ensure angles are sorted to maintain a consistent "convex" ordering
    # This prevents "crossing" in the parameterization if we were using gaps,
    # but here we use absolute angles. Sorting them ensures we always have
    # a valid set of points on the circle.
    # Note: The set of points is invariant to permutation, so sorting is safe.
    thetas = np.sort(thetas)
    
    pts = np.column_stack([np.cos(thetas), np.sin(thetas)])
    
    # Calculate all triangle areas
    areas = []
    for i in range(N):
        p1 = pts[i]
        for j in range(i + 1, N):
            p2 = pts[j]
            v1 = p2 - p1
            for k in range(j + 1, N):
                p3 = pts[k]
                v2 = p3 - p1
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                area = 0.5 * abs(cross)
                areas.append(area)
                
    areas = np.array(areas)
    
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # We want to MAXIMIZE the min area, so we MINIMIZE -min_area.
    # Soft min approximates min.
    # To avoid overflow, shift areas
    max_area = np.max(areas)
    exp_terms = np.exp(-beta * (areas - max_area))
    log_sum_exp = max_area + np.log(np.sum(exp_terms)) / beta
    soft_min_val = log_sum_exp
    
    return -soft_min_val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    # Initial guess: Regular N-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    # We optimize angles[1:]
    x0 = angles[1:].copy()
    
    best_score = -float('inf')
    best_x = x0
    
    # Strategy:
    # 1. Global-ish search with random restarts + local polish
    # 2. Use soft-min for gradient-based optimization (L-BFGS-B)
    # 3. Polish with Powell on exact min
    
    num_restarts = 10
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        if r == 0:
            current_x = x0
        else:
            # Perturb regular polygon
            current_x = x0 + np.random.normal(0, 0.1, size=N-1)
            # Normalize angles to [0, 2pi) and sort
            current_x = current_x % (2 * np.pi)
            current_x = np.sort(current_x)
            
        # Phase 1: L-BFGS-B with soft min
        # Start with low beta, increase it
        betas = [1.0, 5.0, 10.0, 20.0, 50.0]
        
        for beta in betas:
            if time.time() > deadline:
                break
                
            try:
                res = minimize(
                    _soft_min_objective, 
                    current_x, 
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'disp': False}
                )
                current_x = res.x
            except Exception:
                pass

        # Phase 2: Powell on exact min
        def exact_obj(theta_vars):
            thetas = np.zeros(N)
            thetas[0] = 0.0
            thetas[1:] = theta_vars
            thetas = np.sort(thetas)
            pts = np.column_stack([np.cos(thetas), np.sin(thetas)])
            return -_min_triangle_area(pts)
            
        try:
            res_p = minimize(
                exact_obj, 
                current_x, 
                method='Powell',
                options={'maxiter': 100, 'disp': False}
            )
            current_x = res_p.x
        except Exception:
            pass
        
        # Evaluate final score
        thetas = np.zeros(N)
        thetas[0] = 0.0
        thetas[1:] = current_x
        thetas = np.sort(thetas)
        pts = np.column_stack([np.cos(thetas), np.sin(thetas)])
        
        score = _score(pts)
        
        if score > best_score:
            best_score = score
            best_x = current_x
    
    # Construct final points from best_x
    thetas = np.zeros(N)
    thetas[0] = 0.0
    thetas[1:] = best_x
    thetas = np.sort(thetas)
    
    pts = np.column_stack([np.cos(thetas), np.sin(thetas)])
    
    return pts
# EVOLVE-BLOCK-END
