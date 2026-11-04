import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools

N = 13

# EVOLVE-BLOCK-START
def _min_triangle_area(points):
    """Calculate the minimum area of any triangle formed by 3 points."""
    n = len(points)
    min_area = float('inf')
    # Vectorized approach for speed
    # Create indices for all triples
    # C(13,3) = 286, so a simple loop is fine and fast enough
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area = 0.5 * | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _score(points):
    """Calculate the score: min triangle area / convex hull area."""
    points = np.asarray(points)
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
        if hull_area < 1e-10:
            return 0.0
    except Exception:
        return 0.0
    
    min_area = _min_triangle_area(points)
    if min_area == 0:
        return 0.0
        
    return min_area / hull_area

def _objective(params):
    """Objective function for minimization (negative score)."""
    points = params.reshape(-1, 2)
    s = _score(points)
    # If score is 0, return a large positive value to minimize
    if s == 0:
        return 1.0
    return -s

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Initial guess: Regular 13-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    initial_points = np.column_stack((np.cos(angles), np.sin(angles)))
    initial_params = initial_points.flatten()
    
    best_params = initial_params.copy()
    best_score = _score(initial_points)
    
    # Multi-start optimization
    # We will try a few different starting points and solvers
    
    # 1. Optimize from regular polygon using L-BFGS-B
    try:
        res = minimize(_objective, initial_params, method='L-BFGS-B', 
                       options={'maxiter': 500, 'ftol': 1e-8})
        if res.success and res.fun < 0: # Negative fun means positive score
            curr_score = -res.fun
            if curr_score > best_score:
                best_score = curr_score
                best_params = res.x
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_params.reshape(-1, 2)
        
    # 2. Try Nelder-Mead on the best result so far
    try:
        res2 = minimize(_objective, best_params, method='Nelder-Mead', 
                        options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
        if res2.success and res2.fun < 0:
            curr_score = -res2.fun
            if curr_score > best_score:
                best_score = curr_score
                best_params = res2.x
    except Exception:
        pass

    if time.time() > deadline:
        return best_params.reshape(-1, 2)

    # 3. Try a few random perturbations of the regular polygon to escape local minima
    rng = np.random.default_rng(42)
    for i in range(10):
        if time.time() > deadline:
            break
            
        # Perturb angles and radii
        angles_pert = angles + rng.normal(0, 0.1, N)
        radii_pert = 1.0 + rng.normal(0, 0.1, N)
        radii_pert = np.clip(radii_pert, 0.5, 1.5)
        
        # Ensure angles are sorted to maintain convexity/ordering? 
        # Actually, L-BFGS-B on x,y doesn't care about ordering, but ConvexHull does.
        # Random perturbations of x,y directly:
        points_pert = initial_points + rng.normal(0, 0.1, (N, 2))
        params_pert = points_pert.flatten()
        
        try:
            res3 = minimize(_objective, params_pert, method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-6})
            if res3.success and res3.fun < 0:
                curr_score = -res3.fun
                if curr_score > best_score:
                    best_score = curr_score
                    best_params = res3.x
        except Exception:
            pass

    # Final check
    if time.time() > deadline:
        return best_params.reshape(-1, 2)

    # One last polish with Nelder-Mead
    try:
        res4 = minimize(_objective, best_params, method='Nelder-Mead', 
                        options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
        if res4.success and res4.fun < 0:
            curr_score = -res4.fun
            if curr_score > best_score:
                best_params = res4.x
    except Exception:
        pass

    return best_params.reshape(-1, 2)
# EVOLVE-BLOCK-END
