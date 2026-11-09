# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13

    def triangle_area(p1, p2, p3):
        return abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])) / 2.0

    def min_triangle_area(points):
        """Compute the minimum triangle area among all triples of points."""
        min_area = np.inf
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    area = triangle_area(points[i], points[j], points[k])
                    if area < min_area:
                        min_area = area
        return min_area

    def hull_area(points):
        """Compute the area of the convex hull of the points."""
        try:
            hull = ConvexHull(points)
            return hull.volume
        except Exception:
            return 0.0

    def score(points):
        """Score = A_min / A_hull. Higher is better."""
        a_hull = hull_area(points)
        if a_hull <= 0:
            return 0.0
        a_min = min_triangle_area(points)
        return a_min / a_hull

    def negative_score(points_flat):
        """Negative score for minimization."""
        points = points_flat.reshape(n, 2)
        # Check for NaN or inf
        if not np.all(np.isfinite(points)):
            return 1e10
        s = score(points)
        return -s

    # Start with a good initial configuration: regular polygon + center + some perturbation
    # For 13 points, use a regular 12-gon plus center, then optimize
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    initial = np.column_stack([np.cos(angles), np.sin(angles)])
    # Add center point
    initial = np.vstack([initial, np.array([0.0, 0.0])])
    # Shuffle to randomize order
    np.random.seed(42)
    idx = np.random.permutation(n)
    initial = initial[idx]

    # Normalize to unit area hull
    hull_area_init = hull_area(initial)
    if hull_area_init > 0:
        initial = initial / np.sqrt(hull_area_init)

    # Use Nelder-Mead optimization
    result = minimize(
        negative_score,
        initial.flatten(),
        method='Nelder-Mead',
        options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
    )

    points_opt = result.x.reshape(n, 2)

    # Verify validity
    if not np.all(np.isfinite(points_opt)):
        # Fallback to regular polygon + center
        angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        points_opt = np.column_stack([np.cos(angles), np.sin(angles)])
        points_opt = np.vstack([points_opt, np.array([0.0, 0.0])])
        return points_opt

    # Check that convex hull works
    try:
        ConvexHull(points_opt)
    except Exception:
        # Fallback
        angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        points_opt = np.column_stack([np.cos(angles), np.sin(angles)])
        points_opt = np.vstack([points_opt, np.array([0.0, 0.0])])
        return points_opt

    return points_opt


# EVOLVE-BLOCK-END
