"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

def _min_triangle_area(points):
    """Compute the minimum area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    # Vectorized approach for speed if needed, but N=14 is small (364 triangles)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10

def _score(points):
    """Score: minimum triangle area divided by convex hull area."""
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-10:
        return 0.0
    return min_area / hull_area

def _regular_polygon(n, radius=1.0):
    """Generate points of a regular n-gon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return np.column_stack((x, y))

def construct():
    """
    Optimize point positions starting from a regular polygon configuration
    using local search (Nelder-Mead).
    """
    best_points = None
    best_score = -1.0
    
    # Start with a regular 14-gon
    initial_points = _regular_polygon(N, radius=1.0)
    
    # Define objective for minimization (we maximize score)
    def objective(x):
        points = x.reshape(N, 2)
        return -_score(points)
    
    # Run Nelder-Mead from the regular polygon
    # We might need to scale the initial simplex appropriately
    # Nelder-Mead default simplex is based on initial point + 5% step
    res = minimize(
        objective, 
        initial_points.flatten(), 
        method='Nelder-Mead',
        options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
    )
    
    if res.fun < -best_score:
        best_score = -res.fun
        best_points = res.x.reshape(N, 2)
        
    # Try a second start: Regular polygon with slightly different radius or rotation
    # Or perturb the regular polygon
    rng = np.random.default_rng(42)
    
    for i in range(3):
        # Perturb the regular polygon
        perturbation = rng.normal(0, 0.1, size=(N, 2))
        current_points = initial_points + perturbation
        
        res = minimize(
            objective, 
            current_points.flatten(), 
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        
        if res.fun < -best_score:
            best_score = -res.fun
            best_points = res.x.reshape(N, 2)

    return best_points
# EVOLVE-BLOCK-END
