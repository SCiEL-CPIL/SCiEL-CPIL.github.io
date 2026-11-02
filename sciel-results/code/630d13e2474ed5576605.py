import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull

N = 14

def _min_triangle_area(points):
    """
    Compute the minimum area of any triangle formed by 3 points.
    Vectorized for speed.
    """
    min_area = float('inf')
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    
    # We iterate over pairs (i, j) and compute areas with all k > j
    # This avoids redundant computation
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            
            # Vectors from i to all k > j
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            areas = 0.5 * np.abs(cross)
            
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
                if min_area < 1e-12:
                    return 0.0
                    
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

def _decode_central_symmetric(params):
    """
    Decode 14 parameters into 14 points forming a centrally symmetric convex polygon.
    The first 7 parameters define the first 7 vertices.
    The last 7 vertices are the negatives of the first 7 (rotated by pi).
    
    To ensure convexity, we define the points in polar coordinates relative to the center,
    but enforce that the angles are strictly increasing and wrap around.
    
    Actually, a more robust way to ensure convexity for centrally symmetric polygons:
    Define 7 vectors v_0 ... v_6.
    The full set of 14 vertices is v_0, v_1, ..., v_6, -v_6, -v_5, ..., -v_0.
    For this to be a convex polygon, the cross products of consecutive edge vectors must have the same sign.
    
    Let's use a different parameterization:
    Define angles theta_0 < theta_1 < ... < theta_6 < theta_0 + pi.
    And radii r_0 ... r_6.
    The 14 points are:
    P_i = r_i * (cos theta_i, sin theta_i) for i=0..6
    P_{i+7} = r_i * (cos (theta_i + pi), sin (theta_i + pi)) = -P_i
    
    This guarantees central symmetry.
    Convexity is guaranteed if the points are ordered by angle around the center.
    Since theta_i are increasing and r_i > 0, the points P_0..P_6 are in angular order.
    The points -P_6..-P_0 are in angular order as well (angles + pi).
    The transition from P_6 to -P_6 (which is P_13) and from -P_0 (P_7) to P_0 needs care.
    
    Let's just check convexity explicitly and penalize if not convex.
    Or better, use the fact that for a centrally symmetric polygon, convexity is equivalent to
    the 7 points P_0..P_6 forming a "half" polygon that doesn't wrap around the center in a bad way.
    
    Actually, the simplest robust approach:
    Generate 14 points.
    Check if they form a convex hull with 14 vertices.
    If not, return a very bad score.
    """
    # params: 14 floats.
    # First 7 are (r, theta) for i=0..6? No, that's 14 params.
    # Let's use 7 angles and 7 radii.
    # angles must be in (0, pi) and sorted.
    # radii must be positive.
    
    r = np.abs(params[:7]) + 0.1  # Ensure positive radii
    # angles in (0, pi)
    # Use a sigmoid or similar to map to (0, pi)
    # Or just sort and clip.
    raw_angles = params[7:14]
    # Map raw_angles to (0, pi)
    angles = np.pi * (1.0 / (1.0 + np.exp(-raw_angles)))
    # Ensure they are distinct and sorted
    angles = np.sort(angles)
    # Add small jitter to ensure distinctness if needed, but sigmoid + sort usually works
    
    # Construct points
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    points_first_half = np.column_stack([x, y])
    
    # Second half is negative
    points_second_half = -points_first_half
    
    # Interleave? No, the order is P_0, P_1, ..., P_6, P_13, P_12, ..., P_7
    # Wait, P_7 = -P_0, P_8 = -P_1, etc.
    # The angular order of the full set:
    # 0 <= theta_0 < theta_1 < ... < theta_6 < pi
    # Then pi + theta_6 > pi + theta_5 > ... > pi + theta_0 > pi
    # So the order around the circle is:
    # P_0 (theta_0), P_1 (theta_1), ..., P_6 (theta_6), -P_6 (theta_6+pi), -P_5 (theta_5+pi), ..., -P_0 (theta_0+pi)
    
    # Let's construct the full array in this angular order.
    # Indices: 0, 1, 2, 3, 4, 5, 6, 13, 12, 11, 10, 9, 8, 7
    # Point 0..6: points_first_half
    # Point 7: -P_0
    # Point 8: -P_1
    # ...
    # Point 13: -P_6
    
    # So the sequence is:
    # P_0, P_1, P_2, P_3, P_4, P_5, P_6, -P_6, -P_5, -P_4, -P_3, -P_2, -P_1, -P_0
    
    pts = np.vstack([
        points_first_half,
        -points_first_half[::-1]
    ])
    
    return pts

def _is_convex(points):
    """
    Check if points are in convex position.
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

def _objective(params):
    """
    Objective function for optimization.
    params: 14 values.
    """
    points = _decode_central_symmetric(params)
    
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
    
    # Initial guess
    # Regular 14-gon is centrally symmetric.
    # Angles: 0, 2pi/14, 4pi/14, 6pi/14, 8pi/14, 10pi/14, 12pi/14
    # But our parameterization uses angles in (0, pi).
    # So we pick 7 angles in (0, pi) that correspond to the first half of the regular 14-gon.
    # Regular 14-gon angles: k * 2pi/14 = k * pi/7.
    # k=1..6: pi/7, 2pi/7, 3pi/7, 4pi/7, 5pi/7, 6pi/7.
    # These are all in (0, pi).
    
    init_angles = np.pi * np.arange(1, 7) / 7
    init_radii = np.ones(7)
    
    # We need to invert the sigmoid to get raw_angles
    # angles = pi * sigmoid(raw) => sigmoid(raw) = angles/pi => raw = log( (angles/pi) / (1 - angles/pi) )
    ratio = init_angles / np.pi
    raw_angles = np.log(ratio / (1 - ratio))
    
    initial_params = np.concatenate([init_radii, raw_angles])
    
    best_params = initial_params.copy()
    best_score_val = _score(_decode_central_symmetric(initial_params))
    
    # Bounds
    # Radii: [0.5, 3.0]
    # Angles (raw): [-5, 5] (covers sigmoid range well)
    bounds = [(0.5, 3.0)] * 7 + [(-5.0, 5.0)] * 7
    
    # Run Differential Evolution
    try:
        res = differential_evolution(
            _objective, 
            bounds,
            seed=42,
            maxiter=200,
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
                best_params = res.x
    except Exception:
        pass
    
    # Polish with Nelder-Mead
    if time.time() < deadline - 20:
        try:
            res_nm = minimize(
                _objective, 
                best_params, 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if res_nm.fun < _objective(best_params):
                best_params = res_nm.x
                best_score_val = -res_nm.fun
        except Exception:
            pass
            
    # Final Polish with Powell
    if time.time() < deadline - 10:
        try:
            res_powell = minimize(
                _objective, 
                best_params, 
                method='Powell', 
                options={'maxiter': 500, 'xtol': 1e-9, 'ftol': 1e-12}
            )
            if res_powell.fun < _objective(best_params):
                best_params = res_powell.x
                best_score_val = -res_powell.fun
        except Exception:
            pass

    # Convert best params to points
    final_points = _decode_central_symmetric(best_params)
    
    # Normalize to unit area
    try:
        hull = ConvexHull(final_points)
        hull_area = hull.volume
    except Exception:
        # Fallback to regular polygon
        final_points = _decode_central_symmetric(initial_params)
        hull = ConvexHull(final_points)
        hull_area = hull.volume
        
    if hull_area > 1e-10:
        scale = np.sqrt(1.0 / hull_area)
        final_points = final_points * scale
    else:
        final_points = _decode_central_symmetric(initial_params)
        hull = ConvexHull(final_points)
        hull_area = hull.volume
        scale = np.sqrt(1.0 / hull_area)
        final_points = final_points * scale

    return final_points

