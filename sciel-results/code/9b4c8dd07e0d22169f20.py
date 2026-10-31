import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 13

# EVOLVE-BLOCK-START
import random

def _compute_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    if len(points) < 3:
        return 0.0
    
    # Calculate convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume  # In 2D, volume is area
    except Exception:
        # If points are collinear or degenerate, area is 0
        return 0.0
    
    if hull_area < 1e-10:
        return 0.0
        
    # Calculate min triangle area
    min_area = float('inf')
    n = len(points)
    # Vectorized calculation of all triangle areas
    # i, j, k are indices
    # Area = 0.5 * | (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i) |
    
    # To be fast, we can use numpy broadcasting or just a loop since 286 is small
    # Loop is safer and fast enough for 286 iterations
    for i in range(n):
        pi = points[i]
        for j in range(i+1, n):
            pj = points[j]
            v1 = pj - pi
            for k in range(j+1, n):
                pk = points[k]
                v2 = pk - pi
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
                    
    return min_area / hull_area

def _generate_points_from_harmonics(params, n_points=N):
    """
    Generate points from harmonic coefficients.
    params: [a1, b1, a2, b2, ..., aK, bK]
    """
    K = len(params) // 2
    thetas = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    
    r = np.ones(n_points)
    for k in range(1, K + 1):
        a = params[2 * (k - 1)]
        b = params[2 * (k - 1) + 1]
        r += a * np.cos(k * thetas) + b * np.sin(k * thetas)
        
    # Ensure r is positive to avoid self-intersection/origin crossing
    # If r becomes negative or too small, the shape is invalid.
    # We will handle this in the objective function by returning a low score.
    if np.any(r < 0.1): 
        return None

    x = r * np.cos(thetas)
    y = r * np.sin(thetas)
    return np.column_stack((x, y))

def _objective(params):
    """
    Objective function for minimization (we maximize score, so minimize -score).
    """
    points = _generate_points_from_harmonics(params)
    if points is None:
        return 1e10  # Penalty for invalid shape
    
    score = _compute_score(points)
    return -score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Number of harmonics to use
    # Knowledge suggests 4 harmonics (8 params) worked well.
    # Let's try 3 or 4 harmonics.
    K = 4
    n_params = 2 * K
    
    # Initial guess: Regular polygon (all params 0)
    x0 = np.zeros(n_params)
    
    # Bounds for parameters to ensure convexity and simplicity
    # Small amplitudes are safer.
    bounds = [(-0.1, 0.1)] * n_params
    
    best_score = 0.0
    best_points = None
    
    # 1. Try local optimization from regular polygon
    try:
        res = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 200})
        if res.success:
            score = -res.fun
            if score > best_score:
                best_score = score
                best_points = _generate_points_from_harmonics(res.x)
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline:
        if best_points is not None:
            return best_points
        # Fallback to regular polygon
        thetas = np.linspace(0, 2 * np.pi, N, endpoint=False)
        return np.column_stack((np.cos(thetas), np.sin(thetas)))

    # 2. Try differential evolution for global search
    # DE is slower but finds better local optima in non-convex spaces
    try:
        # Use a subset of time for DE
        de_deadline = deadline - 5.0
        
        # Define a wrapper that checks time
        def de_obj(params):
            if time.time() > de_deadline:
                raise TimeoutError()
            return _objective(params)
            
        # Reduce population size for speed
        res_de = differential_evolution(
            de_obj, 
            bounds, 
            maxiter=50, 
            popsize=15, 
            tol=1e-4,
            seed=42,
            polish=False # We will polish later if time permits
        )
        
        if res_de.success:
            score = -res_de.fun
            if score > best_score:
                best_score = score
                best_params = res_de.x
                best_points = _generate_points_from_harmonics(best_params)
            else:
                best_params = res_de.x # Still use DE result for polishing if it's close?
                # Actually, keep the best found so far.
                
    except TimeoutError:
        pass
    except Exception:
        pass

    # 3. Polish the best solution found
    if time.time() < deadline - 5.0 and best_points is not None:
        # Find the params that produced best_points? 
        # We didn't store the params for best_points if it came from L-BFGS.
        # Let's re-run L-BFGS from the best_params found in DE if available, or x0.
        
        # If DE ran, use res_de.x as starting point for final polish
        if 'res_de' in locals() and res_de.success:
            x_start = res_de.x
        else:
            x_start = x0
            
        try:
            res_polish = minimize(_objective, x_start, method='L-BFGS-B', bounds=bounds, options={'maxiter': 100})
            if res_polish.success:
                score = -res_polish.fun
                if score > best_score:
                    best_score = score
                    best_points = _generate_points_from_harmonics(res_polish.x)
        except Exception:
            pass

    if best_points is None:
        # Fallback: Regular polygon
        thetas = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(thetas), np.sin(thetas)))
        
    return best_points
# EVOLVE-BLOCK-END
