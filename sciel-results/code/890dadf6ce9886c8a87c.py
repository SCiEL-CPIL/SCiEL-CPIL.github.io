"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _min_triangle_area(points):
    """Compute minimum triangle area over all triples."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute convex hull area using scipy."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: shoelace on sorted points (approximate)
        # For our use case, points are on a convex curve, so all points are on hull
        # Use shoelace formula on the points in order
        n = len(points)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1] - points[j][0] * points[i][1]
        return abs(area) * 0.5

def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area

def _points_from_fourier(params):
    """
    Generate 14 points from Fourier series parameters.
    params = [a1, b1, a2, b2, scale]
    r(theta) = scale * (1 + a1*cos(theta) + b1*sin(theta) + a2*cos(2*theta) + b2*sin(2*theta))
    """
    a1, b1, a2, b2, scale = params
    thetas = 2 * np.pi * np.arange(N) / N
    r = scale * (1 + a1 * np.cos(thetas) + b1 * np.sin(thetas) + 
                a2 * np.cos(2 * thetas) + b2 * np.sin(2 * thetas))
    # Ensure positive radii
    r = np.maximum(r, 0.1)
    x = r * np.cos(thetas)
    y = r * np.sin(thetas)
    return np.column_stack([x, y])

def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Start with regular 14-gon (scale=1)
    best_params = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    best_points = _points_from_fourier(best_params)
    best_score = _score(best_points)
    
    # Try optimization
    try:
        # Use soft-min surrogate for smoother optimization
        def objective(params):
            points = _points_from_fourier(params)
            # Check convexity roughly: all points should be in convex position
            # For now, just compute score
            min_tri = _min_triangle_area(points)
            hull_area = _convex_hull_area(points)
            if hull_area < 1e-10:
                return -1.0
            return -(min_tri / hull_area)
        
        # Nelder-Mead optimization
        res = minimize(objective, best_params, method='Nelder-Mead',
                      options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8,
                               'adaptive': True})
        
        if res.success and _score(_points_from_fourier(res.x)) > best_score:
            best_params = res.x
            best_points = _points_from_fourier(best_params)
            best_score = _score(best_points)
        
        # Try a few random restarts if time permits
        rng = np.random.RandomState(42)
        for _ in range(5):
            if time.time() > deadline - 30:
                break
            # Random perturbation around best
            params_try = best_params + rng.normal(0, 0.05, size=5)
            # Keep scale reasonable
            params_try[4] = np.clip(params_try[4], 0.5, 2.0)
            
            res2 = minimize(objective, params_try, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-6,
                                   'adaptive': True})
            
            if res2.success:
                score_try = -res2.fun
                if score_try > best_score:
                    best_params = res2.x
                    best_points = _points_from_fourier(best_params)
                    best_score = score_try
        
        # Final polish with tighter tolerance
        if time.time() < deadline - 20:
            res3 = minimize(objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10,
                                   'adaptive': True})
            if res3.success:
                score_try = -res3.fun
                if score_try > best_score:
                    best_params = res3.x
                    best_points = _points_from_fourier(best_params)
                    best_score = score_try
                    
    except Exception:
        pass  # Fall back to best found so far
    
    return best_points
# EVOLVE-BLOCK-END
