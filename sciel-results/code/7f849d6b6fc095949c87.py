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

    def triangle_areas(points):
        """Compute areas of all triangles formed by triples of points."""
        # points: (n, 2)
        # We need to compute area for all i < j < k
        areas = []
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    # Shoelace formula
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    area = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0
                    areas.append(area)
        return np.array(areas)

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
        areas = triangle_areas(points)
        a_min = np.min(areas)
        return a_min / a_hull

    # Initialize with a regular 12-gon + center
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    current_points = np.column_stack([np.cos(angles), np.sin(angles)])
    current_points = np.vstack([current_points, np.array([0.0, 0.0])])
    
    # Normalize to unit hull area
    init_hull = hull_area(current_points)
    if init_hull > 0:
        current_points = current_points / np.sqrt(init_hull)

    current_score = score(current_points)
    best_points = current_points.copy()
    best_score = current_score

    # Simulated Annealing parameters
    T = 0.1
    T_min = 1e-6
    cooling = 0.9995
    steps = 50000
    step_size = 0.1

    np.random.seed(42)

    for i in range(steps):
        # Perturb a random point
        idx = np.random.randint(n)
        perturbation = np.random.normal(0, step_size, 2)
        new_points = current_points.copy()
        new_points[idx] += perturbation

        # Check validity (finite)
        if not np.all(np.isfinite(new_points)):
            continue

        new_score = score(new_points)

        # Acceptance criterion
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = new_points.copy()
        else:
            # Accept with probability exp((new - current) / T)
            if np.random.rand() < np.exp((new_score - current_score) / T):
                current_points = new_points
                current_score = new_score

        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min

    # Final check and return
    if not np.all(np.isfinite(best_points)):
        # Fallback
        angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        points_opt = np.column_stack([np.cos(angles), np.sin(angles)])
        points_opt = np.vstack([points_opt, np.array([0.0, 0.0])])
        return points_opt

    try:
        ConvexHull(best_points)
    except Exception:
        angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        points_opt = np.column_stack([np.cos(angles), np.sin(angles)])
        points_opt = np.vstack([points_opt, np.array([0.0, 0.0])])
        return points_opt

    return best_points


# EVOLVE-BLOCK-END
