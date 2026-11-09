# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.spatial import ConvexHull
from scipy.optimize import minimize


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    time_limit = 600  # Leave 30s margin from 630s

    def triangle_area(p1, p2, p3):
        """Calculate area of triangle formed by three points."""
        return abs(p1[0]*(p2[1] - p3[1]) + p2[0]*(p3[1] - p1[1]) + p3[0]*(p1[1] - p2[1])) / 2.0

    def min_triangle_area(points):
        """Calculate the minimum triangle area among all triples of points."""
        min_area = float('inf')
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    area = triangle_area(points[i], points[j], points[k])
                    if area < min_area:
                        min_area = area
        return min_area

    def hull_area(points):
        """Calculate the convex hull area of the points."""
        try:
            hull = ConvexHull(points)
            return hull.volume
        except Exception:
            return 0.0

    def objective(points):
        """
        Objective function to maximize: A_min / A_hull.
        We minimize the negative of this ratio.
        """
        A_hull = hull_area(points)
        if A_hull <= 0:
            return 0.0
        A_min = min_triangle_area(points)
        return -A_min / A_hull

    def neg_objective(points):
        """Negative of objective for minimization."""
        return objective(points)

    # Start with a reasonable initial configuration
    # Use a regular polygon-like arrangement with some perturbation
    rng = np.random.default_rng(seed=42)
    
    # Try multiple starting configurations
    best_points = None
    best_score = -float('inf')
    
    # Strategy 1: Start with points on a circle (convex position)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    points_circle = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Strategy 2: Start with random points
    points_random = rng.random((n, 2)) * 2 - 1
    
    # Strategy 3: Start with a grid-like arrangement
    points_grid = np.array([
        [-1, -1], [-1, 0], [-1, 1],
        [0, -1], [0, 0], [0, 1],
        [1, -1], [1, 0], [1, 1],
        [-0.5, -0.5], [0.5, -0.5],
        [-0.5, 0.5], [0.5, 0.5]
    ])
    
    initial_configs = [points_circle, points_random, points_grid]
    
    for init_points in initial_configs:
        if time.time() - start_time > time_limit - 60:
            break
        
        # Try Nelder-Mead optimization
        try:
            result = minimize(
                neg_objective,
                init_points,
                method='Nelder-Mead',
                options={
                    'maxiter': 10000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            if result.success or result.fun < 0:
                score = -result.fun
                if score > best_score:
                    best_score = score
                    best_points = result.x.copy()
        except Exception:
            pass
    
    # If no optimization succeeded, use the best initial configuration
    if best_points is None:
        for init_points in initial_configs:
            score = -neg_objective(init_points)
            if score > best_score:
                best_score = score
                best_points = init_points.copy()
    
    # Final check: if still no valid points, return a simple convex configuration
    if best_points is None:
        best_points = points_circle
    
    return best_points


# EVOLVE-BLOCK-END
