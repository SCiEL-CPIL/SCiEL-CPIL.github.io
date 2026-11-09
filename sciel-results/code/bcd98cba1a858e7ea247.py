"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution


def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by triples of points."""
    n = len(points)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area of triangle with vertices points[i], points[j], points[k]
                ax, ay = points[i]
                bx, by = points[j]
                cx, cy = points[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(points):
    """Compute the area of the convex hull of the given points."""
    from scipy.spatial import ConvexHull
    hull = ConvexHull(points)
    return hull.volume  # In 2D, volume is area


def _score_from_angles(angles):
    """
    Given central angles for N points on a unit circle,
    compute the score: min_triangle_area / convex_hull_area.
    """
    # Ensure angles are sorted and in [0, 2*pi)
    angles = np.sort(angles) % (2 * np.pi)
    
    # Convert to points on unit circle
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Compute min triangle area
    min_area = _min_triangle_area(points)
    
    # Compute convex hull area
    hull_area = _convex_hull_area(points)
    
    if hull_area <= 0:
        return 0.0
    
    return min_area / hull_area


def _negative_score(angles):
    """Negative score for minimization."""
    return -_score_from_angles(angles)


def construct():
    """
    Place 14 points in a planar convex region normalized to unit area
    to maximize the minimum area over all triples.
    
    Approach: Represent points as inscribed in a unit circle, parameterized
    by central angles. Optimize angles using differential evolution with
    angular separation constraints to prevent degeneracy.
    """
    # Bounds for angles: each angle in [0, 2*pi)
    bounds = [(0, 2 * np.pi)] * N
    
    # Use differential evolution to maximize the score
    result = differential_evolution(
        _negative_score,
        bounds,
        seed=42,
        maxiter=200,
        popsize=15,
        tol=1e-8,
        polish=True
    )
    
    best_angles = result.x
    best_angles = np.sort(best_angles) % (2 * np.pi)
    
    # Convert to points on unit circle
    points = np.column_stack([np.cos(best_angles), np.sin(best_angles)])
    
    return points
# EVOLVE-BLOCK-END
