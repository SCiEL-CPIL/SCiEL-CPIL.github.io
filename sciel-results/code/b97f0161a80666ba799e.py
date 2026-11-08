"""Heilbronn convex regions: maximize minimum triangle area over all triples."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13

# EVOLVE-BLOCK-START
import itertools

def _convex_hull_area(points):
    """Compute the area of the convex hull of the given points."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: use shoelace formula on all points (may overestimate if not convex)
        return 0.0

def _min_triangle_area(points):
    """Compute the minimum triangle area among all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = points[i]
                b = points[j]
                c = points[k]
                area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _soft_min_objective(params, temperature):
    """
    Soft-min surrogate for the negative log-sum-exp of normalized triangle areas.
    We want to MAXIMIZE the minimum triangle area, so we MINIMIZE the negative soft-min.
    """
    n = len(params) // 2
    points = params.reshape((n, 2))
    
    # Compute convex hull area
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 1e6
    
    # Compute all triangle areas
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = points[i]
                b = points[j]
                c = points[k]
                area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
                areas.append(area)
    
    areas = np.array(areas)
    
    # Normalize by hull area
    normalized_areas = areas / hull_area
    
    # Soft-min: -log(sum(exp(-area/temperature))) * temperature
    # This approximates -min(areas) as temperature -> 0
    # We minimize this, which is equivalent to maximizing the soft-min of areas
    scaled = -normalized_areas / temperature
    # Clip to avoid overflow
    scaled = np.clip(scaled, -100, 100)
    log_sum_exp = np.log(np.sum(np.exp(scaled)))
    soft_min_value = -temperature * log_sum_exp
    
    # We want to maximize soft_min_value, so minimize its negative
    return -soft_min_value

def _gradient(params, temperature):
    """Numerical gradient for the soft-min objective."""
    eps = 1e-7
    n = len(params)
    grad = np.zeros(n)
    f0 = _soft_min_objective(params, temperature)
    for i in range(n):
        params_plus = params.copy()
        params_plus[i] += eps
        f_plus = _soft_min_objective(params_plus, temperature)
        grad[i] = (f_plus - f0) / eps
    return grad

def _regular_polygon(n, radius=1.0):
    """Generate vertices of a regular n-gon centered at origin."""
    angles = 2 * np.pi * np.arange(n) / n
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return np.column_stack((x, y))

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Initialize with regular 13-gon
    initial_points = _regular_polygon(N, radius=1.0)
    initial_params = initial_points.flatten()
    
    best_params = initial_params.copy()
    best_score = _score(initial_points)
    
    # Try different temperatures for soft-min annealing
    temperatures = [0.1, 0.05, 0.02, 0.01, 0.005]
    
    for temp in temperatures:
        if time.time() > deadline - 10:
            break
        
        try:
            result = minimize(
                _soft_min_objective,
                initial_params,
                args=(temp,),
                method='L-BFGS-B',
                jac=_gradient,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < _soft_min_objective(initial_params, temp):
                candidate_points = result.x.reshape((N, 2))
                candidate_score = _score(candidate_points)
                
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_params = result.x.copy()
                    initial_params = result.x.copy()  # Use improved solution for next temperature
        
        except Exception:
            continue
    
    # Final local refinement with Powell (no gradient needed)
    if time.time() < deadline - 5:
        try:
            def _neg_score(params):
                points = params.reshape((N, 2))
                return -_score(points)
            
            result = minimize(
                _neg_score,
                best_params,
                method='Powell',
                options={'maxiter': 100, 'xtol': 1e-10, 'ftol': 1e-12}
            )
            
            if result.fun < -best_score:
                candidate_points = result.x.reshape((N, 2))
                candidate_score = _score(candidate_points)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_params = result.x.copy()
        
        except Exception:
            pass
    
    # Return the best points found
    final_points = best_params.reshape((N, 2))
    return final_points
# EVOLVE-BLOCK-END
