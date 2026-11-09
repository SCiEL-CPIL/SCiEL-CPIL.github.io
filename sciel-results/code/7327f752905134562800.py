import numpy as np
from scipy.spatial import ConvexHull
import time
from scipy.optimize import differential_evolution, minimize

N = 14

# EVOLVE-BLOCK-START

def _min_triangle_area_fast(points):
    """
    Fast calculation of minimum triangle area using vectorized operations.
    Points: (N, 2) array. Returns minimum area (not doubled).
    """
    n = len(points)
    min_area = float('inf')
    for i in range(n - 2):
        pi = points[i]
        for j in range(i + 1, n - 1):
            v1 = points[j] - pi
            v2s = points[j + 1:] - pi
            cross_vals = v1[0] * v2s[:, 1] - v1[1] * v2s[:, 0]
            areas = 0.5 * np.abs(cross_vals)
            if len(areas) > 0:
                cur_min = np.min(areas)
                if cur_min < min_area:
                    min_area = cur_min
    return min_area

def _score(points):
    """
    Score = Min Triangle Area / Convex Hull Area.
    """
    try:
        if np.allclose(points, points[0]):
            return 0.0
        
        hull = ConvexHull(points)
        hull_area = hull.volume
        
        if hull_area < 1e-12:
            return 0.0
            
        min_tri_area = _min_triangle_area_fast(points)
        
        if min_tri_area < 1e-12:
            return 0.0
            
        return min_tri_area / hull_area
    except Exception:
        return 0.0

def _negative_score(points):
    """
    Objective function for minimizer.
    """
    return -_score(points)

def _quadratic_params_to_points(params):
    """
    Convert 3 parameters (width, height, aspect) to 14 points on a parabola.
    x = width * (t^2 - 0.5)
    y = height * (1 - t^2)
    where t is in [-1, 1] uniformly spaced.
    The aspect ratio is applied to x to allow for stretching.
    Actually, let's define:
    t_i = -1 + 2*i/(N-1)
    x_i = width * (t_i^2 - 0.5)  # Range [-0.5w, 0.5w]
    y_i = height * (1 - t_i^2)   # Range [0, h]
    
    To allow more general shapes, we can also add a shear or rotation, 
    but let's stick to symmetric parabola first. 
    Parameters: [width, height, t_offset]
    t_offset shifts the sampling range slightly to break symmetry if needed, 
    or we can just use fixed t. 
    Let's use 3 params: width, height, and a 'flatness' or 'bulge' factor?
    No, a parabola is fixed shape up to scaling. 
    To get more freedom, we can use a higher order polynomial or 
    parameterize the points on the convex hull of a regular polygon 
    but with variable radii.
    
    Wait, the directive says: "Parameterize the 14 points using a convex quadratic form (e.g., x = a0 + a1 t + a2 t^2, y = b0 + b1 t + b2 t^2) where t is uniformly spaced, ensuring all points lie on a smooth convex curve that defines the hull, and optimize the six polynomial coefficients"
    
    So we have 6 coefficients: a0, a1, a2, b0, b1, b2.
    t is uniformly spaced in [0, 1] or [-1, 1].
    Let t = linspace(-1, 1, N).
    x = a0 + a1*t + a2*t^2
    y = b0 + b1*t + b2*t^2
    
    We need to ensure convexity. For a parabola y = ax^2+bx+c, 
    if we map t to x,y, the curve is a parabola. 
    A parabola is convex if it opens "up" or "down" relative to the y-axis, 
    but here we have x and y as functions of t. 
    The curve is a conic section. For it to be convex and enclose area, 
    it should be a parabola segment.
    
    To maximize the score, we want the points to be as "spread out" as possible 
    while keeping the min triangle area high.
    A regular polygon is the best for points on a circle. 
    A parabola is not a closed curve. 
    The convex hull of points on a parabola y=x^2 is a polygon with the 
    two endpoints and the points in between? No, the hull of points on a 
    convex curve is the polygon formed by the points in order.
    
    So the "region" is the convex hull of these 14 points.
    The area of this hull is the area of the polygon.
    
    Let's optimize the 6 coefficients.
    Bounds: 
    a2, b2 should be such that the curve is convex.
    Let's fix the domain t in [-1, 1].
    To avoid degenerate solutions (collinear points), we need curvature.
    
    Let's try to center the shape.
    We can normalize the points after generation to ensure scale invariance 
    is handled by the score, but for optimization stability, 
    we can constrain the coefficients.
    
    Let's use:
    x = a0 + a1*t + a2*t^2
    y = b0 + b1*t + b2*t^2
    
    We want the points to be distinct and non-collinear.
    If a2=0 and b2=0, they are collinear.
    
    Let's define the parameters as:
    p = [a0, a1, a2, b0, b1, b2]
    
    To ensure convexity of the hull (which is always true for points on a convex curve 
    ordered by t, as long as the curve doesn't cross itself), 
    we just need the curve to be simple.
    
    Let's add a constraint or penalty if the area is small.
    
    For the optimizer, we can use Differential Evolution on the 6 parameters.
    """
    a0, a1, a2, b0, b1, b2 = params
    t = np.linspace(-1, 1, N)
    x = a0 + a1 * t + a2 * t**2
    y = b0 + b1 * t + b2 * t**2
    points = np.column_stack([x, y])
    return points

def _score_quadratic_params(params):
    """
    Score for quadratic parameters.
    """
    points = _quadratic_params_to_points(params)
    return _score(points)

def _negative_score_quadratic_params(params):
    """
    Negative score for quadratic parameters (for minimization).
    """
    return -_score_quadratic_params(params)

def _refine_local(params, func, method='Nelder-Mead', maxiter=200, timeout=10):
    """
    Local refinement using Nelder-Mead.
    """
    start_time = time.time()
    def callback(xk):
        if time.time() - start_time > timeout:
            raise KeyboardInterrupt()
    
    try:
        res = minimize(func, params, method=method,
                       options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True},
                       callback=callback)
        if time.time() - start_time > timeout:
            return params
        return res.x
    except KeyboardInterrupt:
        return params
    except Exception:
        return params

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_points = None
    best_score = 0.0
    
    def update_best(points, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_points = points.copy()
            best_score = score

    # Strategy: Optimize 6 quadratic coefficients
    # We expect the optimal shape to be a "wide" parabola or similar.
    # Initial guess: a symmetric parabola y = 1 - x^2 scaled.
    # x = t, y = 1 - t^2.
    # a0=0, a1=1, a2=0
    # b0=1, b1=0, b2=-1
    
    # Let's broaden the search space.
    # Coefficients can be negative.
    # To avoid huge coordinates, we can bound them.
    # But the score is scale invariant. 
    # However, if coordinates are too large, numerical issues might arise.
    # If coordinates are too small, also issues.
    # Let's bound them to [-10, 10] initially.
    
    bounds = [(-10, 10), (-10, 10), (-10, 10), (-10, 10), (-10, 10), (-10, 10)]
    
    # We need to ensure the points are not collinear.
    # DE might find a collinear solution (score 0).
    # We can add a penalty for small hull area.
    
    def _objective_quadratic(params):
        s = _score_quadratic_params(params)
        # If score is 0, it might be degenerate. 
        # DE minimizes, so 0 is "good" if we're not careful? 
        # No, we want to maximize score. 
        # _negative_score returns -score. 
        # So we minimize -score. 
        # If score is 0, -score is 0. 
        # If score is 0.1, -score is -0.1. 
        # So 0 is worse than -0.1. 
        # So DE will try to find negative values (positive scores).
        # Degenerate solutions have score 0, which is the "worst" in the 
        # sense that we want to go lower (more negative).
        # So DE should naturally avoid degenerate solutions if it finds 
        # any valid configuration with score > 0.
        
        return -_score_quadratic_params(params)

    try:
        if time.time() < deadline - 60:
            # Use Sobol initialization for better coverage
            de_res = differential_evolution(
                _objective_quadratic,
                bounds,
                seed=42,
                maxiter=100,
                tol=1e-8,
                popsize=20,
                init='sobol'
            )
            
            best_params_quad = de_res.x
            best_score_quad = -de_res.fun
            best_points_quad = _quadratic_params_to_points(best_params_quad)
            
            # Verify and update
            s = _score(best_points_quad)
            if s > best_score:
                update_best(best_points_quad, s)
            else:
                # DE might have found a local optimum that is worse than 
                # a simple guess? Unlikely, but let's check.
                pass
            
            # Local refinement
            if time.time() < deadline - 30:
                refined_params = _refine_local(best_params_quad, _objective_quadratic, timeout=15)
                refined_points = _quadratic_params_to_points(refined_params)
                s = _score(refined_points)
                if s > best_score:
                    update_best(refined_points, s)
    
    except Exception as e:
        pass

    # Fallback: If quadratic strategy failed or found 0, try regular polygon
    if best_score < 1e-6:
        # Regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        reg_poly = np.column_stack([np.cos(angles), np.sin(angles)])
        s = _score(reg_poly)
        if s > best_score:
            update_best(reg_poly, s)

    # Additional refinement: Try to "unfurl" the parabola into a more polygon-like shape?
    # The quadratic constraint is quite restrictive. 
    # Maybe we should relax it? 
    # The directive asks for this specific strategy. 
    # But if it's bad, we should at least try to improve it.
    
    # Let's try a few random restarts for the quadratic parameters 
    # to escape local minima.
    rng = np.random.default_rng(123)
    for i in range(5):
        if time.time() > deadline - 10:
            break
        
        # Random start
        x0 = rng.uniform(-5, 5, 6)
        
        # Quick local refinement
        refined_params = _refine_local(x0, _objective_quadratic, timeout=5)
        refined_points = _quadratic_params_to_points(refined_params)
        s = _score(refined_points)
        if s > best_score:
            update_best(refined_points, s)

    if best_points is None:
        # Absolute fallback
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points

# EVOLVE-BLOCK-END
