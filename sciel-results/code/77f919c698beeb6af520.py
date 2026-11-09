# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from itertools import combinations

def _triangle_areas(pts):
    """Compute all triangle areas for a set of points. Returns array of areas."""
    n = len(pts)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                x3, y3 = pts[k]
                area = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)) / 2.0
                areas.append(area)
    return np.array(areas)

def _score(points):
    """Compute the Heilbronn score: A_min / A_hull."""
    try:
        hull = ConvexHull(points)
        a_hull = hull.volume  # For 2D, this is the area
        if a_hull <= 0:
            return 0.0
        a_min = np.min(_triangle_areas(points))
        return a_min / a_hull
    except Exception:
        return 0.0

def _negative_score(points_flat):
    """Objective function for minimizer: negative of the Heilbronn score."""
    points = points_flat.reshape(-1, 2)
    return -_score(points)

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    deadline = time.time() + 600.0
    best_points = None
    best_score = 0.0

    # Generate several initial configurations to start from
    initial_configs = []

    # 1. Hexagonal lattice subset
    # Create a hexagonal lattice and pick 13 points
    hex_points = []
    # Center
    hex_points.append([0.0, 0.0])
    # Ring 1: 6 points
    for angle in np.linspace(0, 2*np.pi, 6, endpoint=False):
        hex_points.append([np.cos(angle), np.sin(angle)])
    # Ring 2: 6 more points (every other one to make 13 total)
    for angle in np.linspace(0, 2*np.pi, 12, endpoint=False)[::2]:
        hex_points.append([2*np.cos(angle), 2*np.sin(angle)])
    initial_configs.append(np.array(hex_points))

    # 2. Regular polygon with center
    poly_points = [[0.0, 0.0]]
    for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
        poly_points.append([np.cos(angle), np.sin(angle)])
    initial_configs.append(np.array(poly_points))

    # 3. Triangular lattice subset
    tri_points = []
    # Row 0: 1 point
    tri_points.append([0.0, 0.0])
    # Row 1: 2 points
    tri_points.append([-0.5, np.sqrt(3)/2])
    tri_points.append([0.5, np.sqrt(3)/2])
    # Row 2: 3 points
    for x in [-1.0, 0.0, 1.0]:
        tri_points.append([x, np.sqrt(3)])
    # Row 3: 3 points
    for x in [-1.5, -0.5, 0.5]:
        tri_points.append([x, 1.5*np.sqrt(3)])
    # Row 4: 3 points
    for x in [-1.0, 0.0, 1.0]:
        tri_points.append([x, 2*np.sqrt(3)])
    initial_configs.append(np.array(tri_points))

    # 4. Random points in unit circle
    rng = np.random.default_rng(seed=42)
    for seed in [42, 123, 999]:
        rng = np.random.default_rng(seed=seed)
        angles = np.sort(rng.random(13) * 2 * np.pi)
        radii = np.sqrt(rng.random(13))
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        initial_configs.append(pts)

    # 5. Square grid subset
    grid_pts = []
    for i in range(4):
        for j in range(4):
            if len(grid_pts) < 13:
                grid_pts.append([i/3.0, j/3.0])
    initial_configs.append(np.array(grid_pts))

    # Evaluate initial configurations
    for config in initial_configs:
        if time.time() > deadline - 60:
            break
        s = _score(config)
        if s > best_score:
            best_score = s
            best_points = config.copy()

    # Local optimization from the best initial point
    if best_points is not None and time.time() < deadline - 60:
        x0 = best_points.flatten()
        try:
            res = minimize(_negative_score, x0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8,
                                   'adaptive': True})
            if time.time() < deadline - 30:
                opt_points = res.x.reshape(-1, 2)
                s = _score(opt_points)
                if s > best_score:
                    best_score = s
                    best_points = opt_points.copy()
        except Exception:
            pass

    # Try a few more random restarts if time permits
    if best_points is not None and time.time() < deadline - 120:
        for seed in [7, 13, 23, 42, 99]:
            if time.time() > deadline - 60:
                break
            rng = np.random.default_rng(seed=seed)
            # Perturb the best solution
            perturbed = best_points + rng.normal(0, 0.1, size=best_points.shape)
            x0 = perturbed.flatten()
            try:
                res = minimize(_negative_score, x0, method='Nelder-Mead',
                              options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-7,
                                       'adaptive': True})
                if time.time() < deadline - 30:
                    opt_points = res.x.reshape(-1, 2)
                    s = _score(opt_points)
                    if s > best_score:
                        best_score = s
                        best_points = opt_points.copy()
            except Exception:
                pass

    if best_points is None:
        # Fallback: return a simple configuration
        best_points = np.array([[i/6.0, j/6.0] for i in range(4) for j in range(4)][:13])

    return best_points

# EVOLVE-BLOCK-END
