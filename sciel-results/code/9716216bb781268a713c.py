"""
Heilbronn convex regions n=14
Strategy: Constrained radial optimization anchored to regular 14-gon angles.
Instead of optimizing angles on a fixed circle, we fix the angles to those of a regular 14-gon
and optimize the radial distances r_i. This reduces the dimensionality and leverages the
symmetry of the regular polygon as a strong prior, while allowing radial distortion to
improve the minimum triangle area. We ensure convexity by checking that the points remain
in convex position (no self-intersections, all cross products have the same sign).
"""

import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull

N = 14

def _min_triangle_area(points):
    """
    Compute the minimum area of any triangle formed by 3 points.
    """
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    
    for i in range(N - 2):
        for j in range(i + 1, N - 1):
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            areas = 0.5 * np.abs(cross)
            
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
                
    return min_area

def _score(points):
    """
    Compute the minimum triangle area divided by the convex hull area.
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
    
    if hull_area < 1e-10:
        return 0.0
        
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _is_convex(points):
    """
    Check if points are in convex position (all cross products have the same sign).
    """
    n = len(points)
    sign = 0
    for i in range(n):
        p0 = points[i]
        p1 = points[(i+1)%n]
        p2 = points[(i+2)%n]
        cross = (p1[0]-p0[0])*(p2[1]-p0[1]) - (p1[1]-p0[1])*(p2[0]-p0[0])
        if abs(cross) < 1e-10:
            return False
        if sign == 0:
            sign = 1 if cross > 0 else -1
        elif (cross > 0 and sign < 0) or (cross < 0 and sign > 0):
            return False
    return True

def _points_from_radii(radii):
    """
    Convert radii to points. Angles are fixed at regular 14-gon positions.
    radii: array of N positive values.
    """
    angles = 2 * np.pi * np.arange(N) / N
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack([x, y])

def _radial_objective(radii):
    """
    Objective function for radial optimization.
    radii: N variables in [0.5, 1.5].
    """
    points = _points_from_radii(radii)
    
    # Check convexity
    if not _is_convex(points):
        return 1e10
        
    score = _score(points)
    
    if score < 1e-6:
        return 1e10
        
    return -score

def construct():
    """
    Generate 14 points to maximize the minimum triangle area over convex hull area.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initial guess: Regular polygon (all radii = 1)
    initial_radii = np.ones(N)
    
    best_radii = initial_radii.copy()
    best_score_val = _score(_points_from_radii(initial_radii))
    
    # Bounds for radii: [0.5, 1.5] to allow significant distortion but avoid degeneracy
    bounds = [(0.5, 1.5)] * N
    
    # Run Differential Evolution
    try:
        res = differential_evolution(
            _radial_objective, 
            bounds,
            seed=42,
            maxiter=100,
            popsize=15,
            tol=1e-6,
            mutation=(0.5, 1.5),
            recombination=0.7,
            polish=False,
            init='latinhypercube'
        )
        
        if res.fun < 1e9:
            current_score = -res.fun
            if current_score > best_score_val:
                best_score_val = current_score
                best_radii = res.x
    except Exception:
        pass
    
    # Polish with Nelder-Mead
    if time.time() < deadline - 20:
        try:
            res_nm = minimize(
                _radial_objective, 
                best_radii, 
                method='Nelder-Mead', 
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if res_nm.fun < _radial_objective(best_radii):
                best_radii = res_nm.x
                best_score_val = -res_nm.fun
        except Exception:
            pass
            
    # Final Polish with Powell
    if time.time() < deadline - 10:
        try:
            res_powell = minimize(
                _radial_objective, 
                best_radii, 
                method='Powell', 
                options={'maxiter': 200, 'xtol': 1e-9, 'ftol': 1e-12}
            )
            if res_powell.fun < _radial_objective(best_radii):
                best_radii = res_powell.x
                best_score_val = -res_powell.fun
        except Exception:
            pass

    # Convert best radii to points
    final_points = _points_from_radii(best_radii)
    
    # Normalize to unit area
    try:
        hull = ConvexHull(final_points)
        hull_area = hull.volume
    except Exception:
        # Fallback to regular polygon
        final_points = _points_from_radii(initial_radii)
        hull = ConvexHull(final_points)
        hull_area = hull.volume
        
    if hull_area > 1e-10:
        scale = np.sqrt(1.0 / hull_area)
        final_points = final_points * scale
    else:
        final_points = _points_from_radii(initial_radii)
        hull = ConvexHull(final_points)
        hull_area = hull.volume
        scale = np.sqrt(1.0 / hull_area)
        final_points = final_points * scale

    return final_points
