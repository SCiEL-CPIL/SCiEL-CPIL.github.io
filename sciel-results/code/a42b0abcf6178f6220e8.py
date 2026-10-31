"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _triangle_area(p1, p2, p3):
    """Compute the area of triangle formed by three points."""
    return 0.5 * abs(np.cross(p2 - p1, p3 - p1))


def _convex_hull_area(points):
    """Compute the area of the convex hull of a set of points."""
    # Use scipy if available, otherwise implement a simple convex hull
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: use the fact that for points on a convex curve,
        # the convex hull is the polygon formed by all points in order
        # But this only works if points are in convex position
        # For safety, implement Andrew's monotone chain
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
        points_sorted = sorted(points, key=lambda p: (p[0], p[1]))
        if len(points_sorted) <= 2:
            return 0.0
        
        lower = []
        for p in points_sorted:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        
        upper = []
        for p in reversed(points_sorted):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        
        hull = lower[:-1] + upper[:-1]
        if len(hull) < 3:
            return 0.0
        
        area = 0.0
        for i in range(len(hull)):
            j = (i + 1) % len(hull)
            area += hull[i][0] * hull[j][1] - hull[j][0] * hull[i][1]
        return abs(area) / 2.0


def _score(points):
    """Compute the minimum triangle area divided by convex hull area."""
    n = len(points)
    min_tri_area = float('inf')
    
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_tri_area:
                    min_tri_area = area
    
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    
    return min_tri_area / hull_area


def _generate_points_from_params(params):
    """
    Generate 13 points from parameters.
    params = [r0, r1, theta1, r2, theta2]
    where r0 is base radius, r1, r2 are amplitudes for harmonics 1 and 2,
    theta1, theta2 are phase shifts.
    """
    r0 = params[0]
    r1 = params[1]
    theta1 = params[2]
    r2 = params[3]
    theta2 = params[4]
    
    points = []
    for i in range(N):
        angle = 2 * np.pi * i / N
        # Radius perturbed by low-frequency Fourier series
        r = r0 + r1 * np.cos(angle + theta1) + r2 * np.cos(2 * angle + theta2)
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        points.append([x, y])
    
    return np.array(points)


def _objective(params):
    """Objective function to maximize (we minimize the negative)."""
    points = _generate_points_from_params(params)
    return -_score(points)


def construct():
    """
    Construct 13 points to maximize the minimum triangle area 
    divided by the convex hull area.
    
    Uses a parameterized approach with low-frequency Fourier perturbations
    of a regular 13-gon, optimized via scipy's minimize.
    """
    # Initial guess: regular 13-gon with unit circumradius
    # r0 ~ 1, small perturbations
    initial_params = np.array([1.0, 0.05, 0.0, 0.02, 0.0])
    
    # Bounds for parameters
    # r0: base radius, keep it positive
    # r1, r2: perturbation amplitudes, keep them small to maintain convexity
    bounds = [
        (0.5, 2.0),   # r0
        (-0.3, 0.3),  # r1
        (-np.pi, np.pi),  # theta1
        (-0.3, 0.3),  # r2
        (-np.pi, np.pi),  # theta2
    ]
    
    # Optimize using Nelder-Mead (derivative-free, good for small problems)
    result = minimize(
        _objective,
        initial_params,
        method='Nelder-Mead',
        bounds=bounds,
        options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
    )
    
    best_params = result.x
    points = _generate_points_from_params(best_params)
    
    # Also try a few different initializations to avoid local minima
    best_points = points
    best_score = _score(points)
    
    for r0_init in [0.8, 1.0, 1.2]:
        for r1_init in [0.0, 0.05, -0.05]:
            for r2_init in [0.0, 0.02, -0.02]:
                for theta1_init in [0.0, np.pi/3, 2*np.pi/3]:
                    for theta2_init in [0.0, np.pi/2]:
                        init = np.array([r0_init, r1_init, theta1_init, r2_init, theta2_init])
                        res = minimize(
                            _objective,
                            init,
                            method='Nelder-Mead',
                            bounds=bounds,
                            options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
                        )
                        pts = _generate_points_from_params(res.x)
                        sc = _score(pts)
                        if sc > best_score:
                            best_score = sc
                            best_points = pts
    
    return best_points
# EVOLVE-BLOCK-END
