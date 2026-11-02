"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

def _min_triangle_area(points):
    """
    Compute the minimum area of any triangle formed by 3 points.
    """
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    
    for i in range(N):
        for j in range(i + 1, N):
            if j + 1 >= N:
                continue
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            areas = 0.5 * np.abs(cross)
            
            if len(areas) > 0:
                local_min = np.min(areas)
                if local_min < min_area:
                    min_area = local_min
                    
    return min_area

def _score(points):
    """
    Compute the minimum triangle area divided by the convex hull area.
    """
    try:
        # Check for degeneracy: all points must not be collinear
        # ConvexHull requires at least 3 non-collinear points
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
    
    if hull_area < 1e-10:
        return 0.0
        
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _objective(points_flat):
    """
    Objective function for Cartesian optimization.
    We want to maximize score, so we minimize -score.
    """
    points = points_flat.reshape(N, 2)
    
    # Simple sanity check
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return 1e10
        
    score = _score(points)
    
    if score < 1e-6:
        return 1e10
        
    return -score

def _points_from_angles(angles):
    """
    Convert angles to points on the unit circle.
    Ensures points are in convex position (on the boundary of the convex hull).
    """
    angles = np.sort(angles)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.column_stack([x, y])

def _angle_objective(angles_flat):
    """
    Objective function for angle-based optimization.
    Minimizes -score.
    """
    angles = angles_flat
    points = _points_from_angles(angles)
    score = _score(points)
    
    if score < 1e-6:
        return 1e10
        
    return -score

def construct():
    """
    Generate 14 points in a planar convex region normalized to unit area
    to maximize the minimum area over all triples.
    
    Strategy:
    1. Use angle-based parameterization to enforce convex position.
    2. Use Differential Evolution on the angle space (14 dimensions) to find a good configuration.
    3. Polish with Nelder-Mead.
    4. Normalize to unit area.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Initial angles: regular 14-gon
    initial_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    # Bounds for angles: [0, 2*pi)
    # Note: sorting happens inside _points_from_angles, so the order doesn't matter for the geometry,
    # but it's good to keep them in a reasonable range for the optimizer.
    bounds = [(0, 2 * np.pi)] * N
    
    best_result_angles = initial_angles.copy()
    best_score_val = _score(_points_from_angles(initial_angles))
    
    # Run Differential Evolution on angles
    try:
        res = differential_evolution(
            _angle_objective, 
            bounds,
            seed=42,
            maxiter=100,
            popsize=15,
            tol=1e-5,
            mutation=(0.5, 1.5),
            recombination=0.7,
            polish=False, # We will do our own polish
            init='latinhypercube'
        )
        
        if res.fun < 1e9:
            current_score = -res.fun
            if current_score > best_score_val:
                best_score_val = current_score
                best_result_angles = res.x
    except Exception:
        pass
    
    # Polish with Nelder-Mead on angles
    if time.time() < deadline - 20:
        try:
            res_nm = minimize(
                _angle_objective, 
                best_result_angles, 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if res_nm.fun < _angle_objective(best_result_angles):
                best_result_angles = res_nm.x
                best_score_val = -res_nm.fun
        except Exception:
            pass
            
    # Final Polish with Powell on angles if time remains
    if time.time() < deadline - 10:
        try:
            res_powell = minimize(
                _angle_objective, 
                best_result_angles, 
                method='Powell', 
                options={'maxiter': 100, 'xtol': 1e-9, 'ftol': 1e-12}
            )
            if res_powell.fun < _angle_objective(best_result_angles):
                best_result_angles = res_powell.x
                best_score_val = -res_powell.fun
        except Exception:
            pass

    # Convert best angles to points
    final_points = _points_from_angles(best_result_angles)
    
    # Normalize to unit area
    try:
        hull = ConvexHull(final_points)
        hull_area = hull.volume
    except Exception:
        # Fallback to regular polygon if hull fails
        final_points = _points_from_angles(initial_angles)
        hull = ConvexHull(final_points)
        hull_area = hull.volume
        
    if hull_area > 1e-10:
        scale = np.sqrt(1.0 / hull_area)
        final_points = final_points * scale
    else:
        # Degenerate case, return regular polygon normalized
        final_points = _points_from_angles(initial_angles)
        hull = ConvexHull(final_points)
        hull_area = hull.volume
        scale = np.sqrt(1.0 / hull_area)
        final_points = final_points * scale

    return final_points
# EVOLVE-BLOCK-END
