"""Rewritten program: Fourier-radius parameterization with DE + L-BFGS-B + Powell polish."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize, Bounds
from scipy.spatial import ConvexHull

N = 14
_DEADLINE = 120.0  # wall-clock limit for evaluator
_SAFETY_MARGIN = 15.0  # seconds to return before deadline


def _convex_hull_area(points):
    """Compute the area of the convex hull of a set of 2D points."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 1.0


def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                # Area of triangle using cross product
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    return min_area


def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area


def _fourier_to_points(params, base_radius=1.0):
    """
    Convert Fourier coefficients to 2D points.
    
    params layout:
      - params[0:2]: ellipse semi-axes (a, b) for the base shape
      - params[2:8]: Fourier coefficients for radial modulation
        modes 1 and 2 (cos1, sin1, cos2, sin2, cos2x, sin2x, cos2y, sin2y)
        Actually let's use:
        r(theta) = a * b / sqrt(b^2*cos^2(theta) + a^2*sin^2(theta)) * (1 + sum of Fourier terms)
    
    Simpler: points on an ellipse with radial perturbation.
    Base: regular 14-gon on unit circle, then apply:
      x = a * cos(theta) * (1 + f_r(theta))
      y = b * sin(theta) * (1 + f_r(theta))
    where f_r is a low-order Fourier series.
    """
    a, b = params[0], params[1]
    # Fourier coefficients for radial modulation
    # Modes: 1, 2, 3 (to allow asymmetric squashing)
    # c1, s1, c2, s2, c3, s3
    c1, s1, c2, s2, c3, s3 = params[2:8]
    
    thetas = 2.0 * np.pi * np.arange(N) / N
    # Radial modulation
    f_r = (c1 * np.cos(thetas) + s1 * np.sin(thetas) +
           c2 * np.cos(2 * thetas) + s2 * np.sin(2 * thetas) +
           c3 * np.cos(3 * thetas) + s3 * np.sin(3 * thetas))
    
    # Ellipse base radius
    r_ellipse = 1.0 / np.sqrt((np.cos(thetas) / b) ** 2 + (np.sin(thetas) / a) ** 2)
    # Actually, parametric ellipse: x = a*cos(t), y = b*sin(t)
    # Let's just use parametric form with radial scaling
    x = a * np.cos(thetas) * (1.0 + f_r)
    y = b * np.sin(thetas) * (1.0 + f_r)
    
    return np.column_stack([x, y])


def _objective(params):
    """Negative score for minimization."""
    points = _fourier_to_points(params)
    # Ensure points are not degenerate
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 1e10
    min_tri = _min_triangle_area(points)
    return -min_tri / hull_area


def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE - _SAFETY_MARGIN
    
    # Fourier parameterization: [a, b, c1, s1, c2, s2, c3, s3]
    # a, b: ellipse semi-axes
    # c1..s3: Fourier coefficients for radial modulation
    lower = [0.5, 0.5, -0.3, -0.3, -0.3, -0.3, -0.2, -0.2]
    upper = [1.5, 1.5, 0.3, 0.3, 0.3, 0.3, 0.2, 0.2]
    
    best_score = 0.0
    best_params = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Initial evaluation
    init_points = _fourier_to_points(best_params)
    best_score = _score(init_points)
    
    # Differential Evolution
    try:
        result_de = differential_evolution(
            _objective,
            bounds=list(zip(lower, upper)),
            seed=42,
            maxiter=200,
            popsize=15,
            tol=1e-6,
            mut=(0.5, 1.0),
            recomb=0.7,
            polish=False,
            init='sobol',
            update_interval=10,
            callback=None,
        )
        if time.time() > deadline:
            # Return best so far
            return _fourier_to_points(best_params)
        
        if result_de.fun < -best_score:
            best_score = -result_de.fun
            best_params = result_de.x
    except Exception:
        pass
    
    if time.time() > deadline:
        return _fourier_to_points(best_params)
    
    # L-BFGS-B polish
    try:
        result_lbfgs = minimize(
            _objective,
            best_params,
            method='L-BFGS-B',
            bounds=list(zip(lower, upper)),
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8},
        )
        if time.time() > deadline:
            return _fourier_to_points(best_params)
        
        if result_lbfgs.fun < -best_score:
            best_score = -result_lbfgs.fun
            best_params = result_lbfgs.x
    except Exception:
        pass
    
    if time.time() > deadline:
        return _fourier_to_points(best_params)
    
    # Powell polish for final refinement
    try:
        result_powell = minimize(
            _objective,
            best_params,
            method='Powell',
            bounds=list(zip(lower, upper)),
            options={'maxiter': 200, 'ftol': 1e-12, 'xtol': 1e-10},
        )
        if result_powell.fun < -best_score:
            best_score = -result_powell.fun
            best_params = result_powell.x
    except Exception:
        pass
    
    return _fourier_to_points(best_params)
