import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from itertools import combinations

N = 14

def _polygon_area(points):
    """
    Calculate the area of a simple polygon given by its vertices in order.
    Assumes points are in CCW or CW order.
    """
    n = len(points)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return 0.5 * abs(area)

def _min_triangle_area(points):
    """
    Calculate the minimum area of any triangle formed by 3 points.
    """
    n = len(points)
    min_area = float('inf')
    # Vectorized approach might be faster, but N=14 is small.
    # Number of triangles is 364.
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                # Area = 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
                area = 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
                if area < min_area:
                    min_area = area
    return min_area

def _score(points):
    """
    Score is min_triangle_area / convex_hull_area.
    """
    hull_area = _polygon_area(points)
    if hull_area < 1e-9:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _generate_points(params):
    """
    params: [a, b, n_super, theta_1, ..., theta_7]
    a, b: semi-axes
    n_super: super-ellipse exponent
    theta_i: angles in [0, pi]
    """
    a, b, n_super = params[0], params[1], params[2]
    thetas = params[3:3+7]
    
    # Sort thetas to ensure proper polygon ordering
    thetas = np.sort(thetas)
    
    # Generate points on super-ellipse in first half (y >= 0)
    # Parametric form of super-ellipse:
    # x = a * sign(cos(t)) * |cos(t)|^(2/n)
    # y = b * sign(sin(t)) * |sin(t)|^(2/n)
    # We use t as the angle parameter, but for super-ellipse, the "angle" in polar coords is not t.
    # However, if we define the points by the parameter t, they are ordered by t.
    # To ensure they are on the boundary, we just use the parametric equations.
    # The "angle" theta in the parameter list is actually the parameter t.
    # We restrict t to [0, pi] for the upper half.
    
    t = thetas
    cos_t = np.cos(t)
    sin_t = np.sin(t)
    
    # Handle sign and power
    # x = a * sign(cos_t) * |cos_t|^(2/n)
    # y = b * sign(sin_t) * |sin_t|^(2/n)
    
    # Since t is in [0, pi], sin_t is always >= 0.
    # cos_t can be positive or negative.
    
    pow_exp = 2.0 / n_super
    
    x = a * np.sign(cos_t) * np.abs(cos_t)**pow_exp
    y = b * np.abs(sin_t)**pow_exp  # sin_t is non-negative
    
    # Create the 7 points in the upper half
    points_upper = np.column_stack((x, y))
    
    # Create the 7 points in the lower half (reflection)
    points_lower = -points_upper
    
    # Combine
    # We need to order them correctly for the polygon area calculation.
    # The points_upper are ordered by t (0 to pi), which corresponds to right-to-left along the top.
    # The points_lower are ordered by t (0 to pi), which corresponds to right-to-left along the bottom.
    # To form a closed loop:
    # Start at rightmost upper point?
    # Let's just concatenate and let the polygon area function handle it, 
    # BUT the polygon area function requires vertices in order around the perimeter.
    
    # Upper points: t=0 (right), t=pi (left).
    # Lower points: t=0 (right), t=pi (left).
    # To go around the perimeter:
    # 1. Upper points from t=0 to t=pi (Right to Left)
    # 2. Lower points from t=pi to t=0 (Left to Right)
    
    # points_upper is sorted by t ascending. So index 0 is t_min, index 6 is t_max.
    # We want to traverse upper from t_max to t_min? Or t_min to t_max?
    # Let's check the geometry.
    # t=0: (a, 0). t=pi/2: (0, b). t=pi: (-a, 0).
    # So points_upper[0] is Right, points_upper[-1] is Left.
    # This is a counter-clockwise traversal of the top half?
    # Right -> Top -> Left. Yes, CCW.
    
    # Now for the bottom half.
    # We want to go from Left (-a, 0) to Right (a, 0) via the bottom.
    # The reflection of points_upper[i] is at the same x, negative y.
    # If we take points_lower in reverse order (t=pi to t=0), we go from Left to Right.
    # points_lower is the reflection of points_upper.
    # points_lower[0] is reflection of points_upper[0] (Right).
    # points_lower[-1] is reflection of points_upper[-1] (Left).
    # So we need points_lower[::-1] to go from Left to Right.
    
    full_points = np.vstack((points_upper, points_lower[::-1]))
    
    return full_points

def _objective(params):
    # params: [a, b, n_super, t1, ..., t7]
    # Bounds for a, b: [0.1, 10.0]
    # Bounds for n_super: [0.5, 10.0] (n=2 is ellipse, n=1 is rhombus, n->inf is rectangle)
    # Bounds for t: [0, pi]
    
    try:
        points = _generate_points(params)
        score = _score(points)
        return -score # Minimize negative score
    except Exception:
        return 0.0

def construct():
    start_time = time.time()
    
    # Define bounds for DE
    # a, b: [0.1, 10]
    # n_super: [0.5, 5.0]
    # t1..t7: [0, pi]
    
    bounds = [
        (0.1, 10.0), # a
        (0.1, 10.0), # b
        (0.5, 5.0),  # n_super
    ] + [(0.0, np.pi)] * 7
    
    # Initial population size
    # DE is stochastic.
    
    best_score = 0.0
    best_points = None
    
    # Run Differential Evolution
    # maxiter and popsize control the effort.
    # 120s limit.
    # DE with 10 parameters.
    # popsize=10, maxiter=50 is a reasonable start.
    
    try:
        result = differential_evolution(
            _objective, 
            bounds, 
            maxiter=100, 
            popsize=15, 
            tol=1e-5, 
            seed=42,
            polish=False # We will polish manually to save time if needed
        )
        
        elapsed = time.time() - start_time
        if elapsed < 100: # If we have time
            # Polish with Nelder-Mead
            x0 = result.x
            res_polish = minimize(_objective, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
            if res_polish.fun < result.fun:
                result = res_polish
                
        if result.fun < 0:
            best_score = -result.fun
            best_params = result.x
            best_points = _generate_points(best_params)
            
    except Exception:
        pass
        
    # If DE failed or didn't find a good score, try a fallback: Regular 14-gon
    # Regular 14-gon score is a known baseline.
    # Let's calculate the regular 14-gon score to have a floor.
    
    if best_points is None or best_score < 0.01:
        # Generate regular 14-gon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        points = np.column_stack((np.cos(angles), np.sin(angles)))
        score = _score(points)
        if score > best_score:
            best_score = score
            best_points = points
    else:
        # Check if regular 14-gon is better (unlikely, but safe)
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        points_reg = np.column_stack((np.cos(angles), np.sin(angles)))
        score_reg = _score(points_reg)
        if score_reg > best_score:
            best_points = points_reg
            best_score = score_reg

    return best_points

# EVOLVE-BLOCK-END
