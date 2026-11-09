import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
import itertools

N = 14

# Precompute triple indices for N=14
_i_list, _j_list, _k_list = [], [], []
for _i in range(N):
    for _j in range(_i + 1, N):
        for _k in range(_j + 1, N):
            _i_list.append(_i)
            _j_list.append(_j)
            _k_list.append(_k)
_TRIPLE_I = np.array(_i_list)
_TRIPLE_J = np.array(_j_list)
_TRIPLE_K = np.array(_k_list)

# Precompute t values for quadratic parameterization
# t is uniformly spaced in [-1, 1]
_T_VALUES = np.linspace(-1, 1, N)
_T2 = _T_VALUES ** 2

def get_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: np.array of shape (N, 2)
    """
    # 1. Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-10:
            return 0.0
    except Exception:
        return 0.0

    # 2. Min Triangle Area (Vectorized)
    p1 = points[_TRIPLE_I]
    p2 = points[_TRIPLE_J]
    p3 = points[_TRIPLE_K]

    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)

    min_area = np.min(areas)
    if min_area < 1e-10:
        return 0.0

    return min_area / hull_area

def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 150s. We stop searching at 140s to be safe.
    deadline = start_time + 140.0

    # Objective function for minimizers (maximize score -> minimize -score)
    def objective(flat_points):
        pts = flat_points.reshape(N, 2)
        return -get_score(pts)

    # Quadratic parameterization: x = a0 + a1*t + a2*t^2, y = b0 + b1*t + b2*t^2
    # This ensures points lie on a smooth convex curve (parabola-like)
    # The 6 coefficients define the curve, and t values are fixed
    
    def quad_to_points(coeffs):
        """Convert 6 quadratic coefficients to N points"""
        a0, a1, a2, b0, b1, b2 = coeffs
        x = a0 + a1 * _T_VALUES + a2 * _T2
        y = b0 + b1 * _T_VALUES + b2 * _T2
        return np.column_stack((x, y))

    def quad_objective(coeffs):
        """Objective for quadratic parameterization"""
        pts = quad_to_points(coeffs)
        return -get_score(pts)

    best_points = None
    best_score = 0.0

    # Strategy:
    # 1. Optimize using quadratic parameterization (6 DOF instead of 28)
    # 2. This ensures convexity and smoothness by construction
    # 3. Use multiple initializations and local refinement
    
    rng = np.random.default_rng(42)

    # Initializations for quadratic coefficients
    # A regular 14-gon can be approximated by a quadratic curve
    # For a circle-like shape, we want a2 and b2 to be negative (concave down)
    # Let's try various initializations
    
    quad_initializations = []
    
    # 1. Symmetric parabola-like: x = a2*t^2, y = b2*t^2 with a2, b2 > 0
    # This creates a U-shape. We want a convex hull, so let's try:
    # x = 1 - a2*t^2, y = b2*t^2 (ellipse-like)
    quad_initializations.append(np.array([1.0, 0.0, -0.5, 0.0, 0.0, 1.0]))  # Ellipse-like
    quad_initializations.append(np.array([0.0, 0.0, -0.5, 0.0, 1.0, 0.0]))  # Parabola opening left
    quad_initializations.append(np.array([0.0, 0.0, -0.3, 0.0, 0.0, -0.3]))  # Rotated ellipse
    
    # 2. Random perturbations around good initializations
    for base in quad_initializations[:3]:
        for _ in range(10):
            noise = rng.normal(0, 0.1, 6)
            quad_initializations.append(base + noise)
    
    # 3. More diverse random initializations
    for _ in range(20):
        # Random coefficients, but keep them in a reasonable range
        coeffs = rng.uniform(-1, 1, 6)
        # Ensure the curve has some extent
        if np.abs(coeffs[2]) + np.abs(coeffs[5]) < 0.1:
            coeffs[2] += 0.5
        quad_initializations.append(coeffs)

    # Optimize using quadratic parameterization
    for init_coeffs in quad_initializations:
        if time.time() > deadline - 20.0:
            break
            
        try:
            # Use Powell's method for the 6-dimensional quadratic space
            res = minimize(
                quad_objective, 
                init_coeffs, 
                method='Powell', 
                options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 2000, 'maxfev': 50000}
            )
            
            if res.fun < -best_score:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_points = quad_to_points(res.x)
                    
        except Exception:
            pass

    # If we have time, do a final fine-tuning in full 2D space
    if best_points is not None and time.time() < deadline - 10.0:
        try:
            # Refine in full 2D space to escape any quadratic constraint limitations
            res_nm = minimize(
                objective, 
                best_points.flatten(), 
                method='Nelder-Mead', 
                options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 20000, 'maxfev': 200000}
            )
            if res_nm.fun < -best_score:
                best_score = -res_nm.fun
                best_points = res_nm.x.reshape(N, 2)
        except Exception:
            pass

    if best_points is None:
        # Fallback to regular polygon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        x = np.cos(angles)
        y = np.sin(angles)
        best_points = np.column_stack((x, y))

    return best_points
