"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from itertools import combinations

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area of triangle = 0.5 * |cross product|
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute the area of the convex hull of the given points."""
    # Use scipy's ConvexHull
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: simple method using shoelace on sorted points
        # Sort by angle from centroid
        centroid = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
        sorted_points = points[np.argsort(angles)]
        area = 0.0
        n = len(sorted_points)
        for i in range(n):
            j = (i + 1) % n
            area += sorted_points[i][0] * sorted_points[j][1]
            area -= sorted_points[j][0] * sorted_points[i][1]
        return abs(area) / 2.0

def _objective(x):
    """Objective to minimize: negative of (min_triangle_area / hull_area)."""
    points = x.reshape(N, 2)
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 1e10
    return -min_area / hull_area

def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    return min_area / hull_area

def construct():
    # Start with a regular polygon configuration as a good initial guess
    # For N=13, place points on a circle, but we need to optimize
    # Initial guess: regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x0 = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()
    
    # Use Nelder-Mead optimization
    result = minimize(
        _objective,
        x0,
        method='Nelder-Mead',
        options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
    )
    
    best_points = result.x.reshape(N, 2)
    best_score = _score(best_points)
    
    # Try a few different starting points to ensure we find a good solution
    for trial in range(5):
        np.random.seed(42 + trial)
        x0_trial = np.random.randn(N * 2) * 0.5
        result_trial = minimize(
            _objective,
            x0_trial,
            method='Nelder-Mead',
            options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
        )
        trial_points = result_trial.x.reshape(N, 2)
        trial_score = _score(trial_points)
        if trial_score > best_score:
            best_points = trial_points
            best_score = trial_score
    
    return best_points
# EVOLVE-BLOCK-END
