"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution
from scipy.spatial import ConvexHull


def _min_triangle_area(points):
    """Compute the minimum area over all triples of points."""
    n = len(points)
    min_area = float('inf')
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
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0


def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area


def _angular_to_points(angles, radii):
    """Convert angular parameters to 2D points on a unit circle."""
    angles = np.asarray(angles)
    radii = np.asarray(radii)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack([x, y])


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with regular polygon
    angles_init = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radii_init = np.ones(N)
    points_init = _angular_to_points(angles_init, radii_init)
    best_points = points_init.copy()
    best_score = _score(best_points)
    
    # Differential evolution over angular parameters
    # We parameterize points by (angle, radius) pairs
    # But to keep it simpler and more robust, let's use 2N coordinates
    # with bounds in a square, then normalize
    
    def objective(params):
        if time.time() > deadline:
            return 0.0
        points = np.reshape(params, (N, 2))
        # Normalize to unit area convex hull
        hull_area = _convex_hull_area(points)
        if hull_area < 1e-10:
            return 0.0
        points_norm = points / np.sqrt(hull_area)
        return _score(points_norm)
    
    # Bounds: each coordinate in [-1, 1]
    bounds = [(-1, 1)] * (2 * N)
    
    try:
        result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=100,
            popsize=15,
            tol=1e-6,
            polish=True,
            init='sobol',
            update_interval=1
        )
        
        if result.fun > best_score:
            best_points = np.reshape(result.x, (N, 2))
            best_score = result.fun
    except Exception:
        pass
    
    # Try a few random restarts if time permits
    rng = np.random.default_rng(42)
    while time.time() < deadline - 5.0:
        # Random points in unit square
        points_rand = rng.random((N, 2)) * 2 - 1
        hull_area = _convex_hull_area(points_rand)
        if hull_area < 1e-10:
            continue
        points_norm = points_rand / np.sqrt(hull_area)
        score = _score(points_norm)
        if score > best_score:
            best_points = points_norm.copy()
            best_score = score
    
    # Normalize final result to unit area convex hull
    hull_area = _convex_hull_area(best_points)
    if hull_area > 1e-10:
        best_points = best_points / np.sqrt(hull_area)
    
    return best_points
# EVOLVE-BLOCK-END
