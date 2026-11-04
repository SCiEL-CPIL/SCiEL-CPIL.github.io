"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

# EVOLVE-BLOCK-START
def _convex_hull_area(points):
    """Compute the area of the convex hull of a set of 2D points."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: use shoelace on all points sorted by angle
        center = points.mean(axis=0)
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        sorted_pts = points[np.argsort(angles)]
        x = sorted_pts[:, 0]
        y = sorted_pts[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    return min_area


def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area


def _soft_min_triangle_area(points, beta=100.0):
    """
    Soft-min surrogate for the minimum triangle area using log-sum-exp.
    Returns a smooth approximation of the min triangle area.
    """
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                areas.append(area)
    areas = np.array(areas)
    # Soft min via log-sum-exp: soft_min(x) = -1/beta * log(sum(exp(-beta*x)))
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return -log_sum_exp / beta


def _objective(params, beta=100.0):
    """
    Objective function for optimization.
    We want to maximize min_triangle_area / hull_area.
    So we minimize -min_triangle_area / hull_area.
    Using soft-min for smoothness.
    """
    points = params.reshape(-1, 2)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 1e10
    soft_min = _soft_min_triangle_area(points, beta=beta)
    return -soft_min / hull_area


def _initial_ring_config():
    """Create an initial configuration with points on a ring (circle)."""
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    return points.flatten()


def _random_config(rng):
    """Create a random configuration."""
    return rng.uniform(-1, 1, size=(N, 2)).flatten()


def construct():
    """
    Construct 14 points to maximize the minimum triangle area divided by convex hull area.
    Uses a two-stage optimization: first with a lower beta (smoother), then higher beta (sharper).
    """
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit

    best_points = None
    best_score = 0.0

    # Stage 1: Coarse optimization with lower beta
    rng = np.random.RandomState(42)

    # Try multiple starting points
    starts = []
    # Ring configuration
    starts.append(_initial_ring_config())
    # Random configurations
    for _ in range(5):
        starts.append(_random_config(rng))

    for beta in [50.0, 250.0]:
        for start in starts:
            if time.time() > deadline:
                break

            try:
                result = minimize(
                    _objective,
                    start,
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
                )
            except Exception:
                continue

            if time.time() > deadline:
                break

            points = result.x.reshape(-1, 2)
            score = _score(points)

            if score > best_score:
                best_score = score
                best_points = points.copy()

        if time.time() > deadline:
            break

    # Final polish: if we have a good candidate, do a high-precision pass
    if best_points is not None and time.time() < deadline - 5.0:
        try:
            result = minimize(
                _objective,
                best_points.flatten(),
                args=(500.0,),
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            points = result.x.reshape(-1, 2)
            score = _score(points)
            if score > best_score:
                best_score = score
                best_points = points.copy()
        except Exception:
            pass

    if best_points is None:
        # Fallback to random
        best_points = np.array([[rng.uniform(-1, 1), rng.uniform(-1, 1)] for _ in range(N)])

    return best_points
# EVOLVE-BLOCK-END
