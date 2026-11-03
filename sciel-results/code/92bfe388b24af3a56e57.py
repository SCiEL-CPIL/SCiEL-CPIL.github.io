"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize


def _min_triangle_area_over_hull(points):
    """
    Compute min triangle area / convex hull area for given points.
    Returns the ratio (higher is better).
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    # Compute convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0
    
    if hull_area <= 1e-10:
        return 0.0
    
    # Vectorized triangle area calculation
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
                    if min_area < 1e-10:
                        return 0.0
    
    return min_area / hull_area


def _objective_function(x):
    """
    Objective function for minimization.
    We minimize the negative of the score.
    """
    points = x.reshape(-1, 2)
    score = _min_triangle_area_over_hull(points)
    return -score


def construct():
    """
    Uses Nelder-Mead optimization to maximize the minimum triangle area ratio.
    """
    # Initial configuration: Regular polygon
    angles = 2 * np.pi * np.arange(N) / N
    x0 = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()
    
    # Normalize initial points to have hull area 1.0
    # This helps with numerical stability, although the objective is scale-invariant.
    try:
        hull = ConvexHull(x0.reshape(-1, 2))
        area = hull.volume
        if area > 0:
            x0 /= np.sqrt(area)
    except Exception:
        pass

    # Run Nelder-Mead
    # maxiter is set high enough to ensure convergence but low enough to stay within time limits.
    # Nelder-Mead is typically very fast for N=14 (28 variables).
    result = minimize(
        _objective_function,
        x0,
        method='Nelder-Mead',
        options={
            'xatol': 1e-6,
            'fatol': 1e-8,
            'maxiter': 10000,
            'maxfev': 10000
        }
    )
    
    best_points = result.x.reshape(-1, 2)
    
    # It's possible Nelder-Mead ends up in a slightly worse local minimum or drifts.
    # We can also try a few random restarts if we have time, but Nelder-Mead from a regular polygon
    # is usually very strong for this symmetric problem.
    # Let's add a simple check: if the score is bad, try a perturbed start?
    # For now, we trust the single run from the symmetric start as it's a distinct, fast approach.
    
    return best_points
# EVOLVE-BLOCK-END
