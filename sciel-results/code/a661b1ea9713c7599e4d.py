"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull


def _min_triangle_area(points):
    """Compute the minimum triangle area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                # Area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(points):
    """Compute the area of the convex hull of the given points."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        if len(points) < 3:
            return 0.0
        # Fallback: compute area using shoelace formula
        # Sort by angle around centroid for a rough polygon
        c = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - c[1], points[:, 0] - c[0])
        order = np.argsort(angles)
        pts = points[order]
        n = len(pts)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += pts[i][0] * pts[j][1]
            area -= pts[j][0] * pts[i][1]
        return abs(area) * 0.5


def _score(points):
    """Score = minimum triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area


def _regular_polygon(n):
    """Generate vertices of a regular n-gon on the unit circle."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.column_stack((x, y))


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

    # Start with regular polygon as initial guess
    best_points = _regular_polygon(N)
    best_score = _score(best_points)

    # Try multiple optimization runs with different initial conditions
    rng = np.random.default_rng(42)

    # Define objective: minimize negative score
    def objective(x):
        pts = x.reshape(-1, 2)
        hull_area = _convex_hull_area(pts)
        if hull_area < 1e-10:
            return 1e10
        min_area = _min_triangle_area(pts)
        return -min_area / hull_area

    # Phase 1: Nelder-Mead from multiple random perturbations of regular polygon
    for restart in range(15):
        if time.time() > deadline:
            break

        if restart == 0:
            points0 = best_points.copy()
        else:
            # Random perturbation of the regular polygon
            points0 = _regular_polygon(N)
            points0 += rng.normal(0, 0.1, size=points0.shape)

        res = minimize(objective, points0.flatten(), method='Nelder-Mead',
                       options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10})

        if time.time() > deadline:
            break

        candidate = res.x.reshape(-1, 2)
        cand_score = _score(candidate)
        if cand_score > best_score:
            best_score = cand_score
            best_points = candidate

    # Phase 2: Use BFGS with gradient-free approach for further refinement
    # Use the best point from Phase 1 as starting point
    for restart in range(5):
        if time.time() > deadline:
            break

        points0 = best_points.copy()
        if restart > 0:
            points0 += rng.normal(0, 0.01, size=points0.shape)

        res = minimize(objective, points0.flatten(), method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12,
                                'adaptive': True})

        if time.time() > deadline:
            break

        candidate = res.x.reshape(-1, 2)
        cand_score = _score(candidate)
        if cand_score > best_score:
            best_score = cand_score
            best_points = candidate

    return best_points
# EVOLVE-BLOCK-END
