import numpy as np
from scipy.optimize import differential_evolution, basinhopping
from scipy.spatial import ConvexHull
import time

N = 14

def _min_triangle_area(points):
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except:
        return 0.0

def _score(points):
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-9:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area

def _fourier_points(params):
    # params: [a1, b1, a2, b2, a3, b3]
    a1, b1, a2, b2, a3, b3 = params
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    r = np.ones(N)
    r += a1 * np.cos(angles) + b1 * np.sin(angles)
    r += a2 * np.cos(2 * angles) + b2 * np.sin(2 * angles)
    r += a3 * np.cos(3 * angles) + b3 * np.sin(3 * angles)
    
    # Ensure r is positive to avoid crossing origin
    r = np.maximum(r, 0.1)
    
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    return np.column_stack((x, y))

def _objective(params):
    points = _fourier_points(params)
    s = _score(points)
    return -s

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s safety margin
    
    # Initial guess: Regular polygon (all coeffs 0)
    x0 = np.zeros(6)
    
    # Bounds for coefficients
    bounds = [(-0.3, 0.3)] * 6
    
    best_points = _fourier_points(x0)
    best_score = _score(best_points)
    
    # 1. Differential Evolution
    if time.time() < deadline - 30:
        try:
            res = differential_evolution(
                _objective, 
                bounds, 
                seed=42, 
                maxiter=200, 
                popsize=20,
                tol=1e-8
            )
            if res.fun < _objective(x0): # If improved
                curr_points = _fourier_points(res.x)
                curr_score = _score(curr_points)
                if curr_score > best_score:
                    best_points = curr_points
                    best_score = curr_score
        except Exception:
            pass

    # 2. Basinhopping for refinement
    if time.time() < deadline - 10:
        try:
            # Start from the best point found so far
            x_start = res.x if 'res' in locals() and res.success else x0
            if not (hasattr(res, 'success') and res.success):
                 # If DE didn't finish or fail, use x0
                 x_start = x0 if 'res' not in locals() else res.x
            
            def local_minimize(x):
                # Use Nelder-Mead for local refinement
                from scipy.optimize import minimize
                res_lm = minimize(_objective, x, method='Nelder-Mead', options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 200})
                return res_lm.x, res_lm.fun, {}
                
            res_bh = basinhopping(
                _objective, 
                x_start, 
                minimizer_kwargs=dict(method='Nelder-Mead', options=dict(xatol=1e-8, fatol=1e-8, maxiter=200)),
                niter=50, 
                stepsize=0.1,
                seed=42
            )
            
            curr_points = _fourier_points(res_bh.x)
            curr_score = _score(curr_points)
            if curr_score > best_score:
                best_points = curr_points
                best_score = curr_score
        except Exception:
            pass
            
    return best_points
