"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14

# Precompute triple indices for N=14
TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)
])

# EVOLVE-BLOCK-START
def _convex_hull_area(points):
    """Compute the area of the convex hull of a set of 2D points."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        center = points.mean(axis=0)
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        sorted_pts = points[np.argsort(angles)]
        x = sorted_pts[:, 0]
        y = sorted_pts[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _all_triangle_areas(points):
    """Vectorized computation of all triangle areas."""
    p1 = points[TRIPLE_INDICES[:, 0]]
    p2 = points[TRIPLE_INDICES[:, 1]]
    p3 = points[TRIPLE_INDICES[:, 2]]
    areas = 0.5 * np.abs((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) -
                         (p3[:, 0] - p1[:, 0]) * (p2[:, 1] - p1[:, 1]))
    return areas


def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by triples of points."""
    return np.min(_all_triangle_areas(points))


def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area


def _soft_min_score(params, beta=100.0):
    """
    Soft-min surrogate for optimization.
    Returns negative score (to be minimized).
    """
    points = params.reshape(-1, 2)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 1e10
    
    areas = _all_triangle_areas(points)
    # Soft min via log-sum-exp
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum_exp / beta
    
    return -soft_min / hull_area


def _two_ring_points(params):
    """
    Generate points from two-ring parameterization.
    params: [r1, phi1_offset, r2, phi2_offset, n1, n2]
    n1 + n2 = N, points on ring 1 at radius r1, ring 2 at radius r2
    """
    r1 = params[0]
    phi1_off = params[1]
    r2 = params[2]
    phi2_off = params[3]
    n1 = int(round(params[4]))
    n2 = N - n1
    
    if n1 < 1 or n2 < 1:
        # Fallback: all on one ring
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + phi1_off
        return np.column_stack([r1 * np.cos(angles), r1 * np.sin(angles)])
    
    angles1 = np.linspace(0, 2 * np.pi, n1, endpoint=False) + phi1_off
    angles2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + phi2_off
    
    pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
    pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
    
    return np.vstack([pts1, pts2])


def _two_ring_score(params):
    """Score for two-ring parameterization."""
    points = _two_ring_points(params)
    return _score(points)


def _three_ring_points(params):
    """
    Generate points from three-ring parameterization.
    params: [r1, phi1, r2, phi2, r3, phi3, n1, n2, n3]
    n1 + n2 + n3 = N
    """
    r1, phi1 = params[0], params[1]
    r2, phi2 = params[2], params[3]
    r3, phi3 = params[4], params[5]
    n1 = int(round(params[6]))
    n2 = int(round(params[7]))
    n3 = N - n1 - n2
    
    if n1 < 1 or n2 < 1 or n3 < 1:
        # Fallback to two-ring
        return _two_ring_points(params[:6])
    
    angles1 = np.linspace(0, 2 * np.pi, n1, endpoint=False) + phi1
    angles2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + phi2
    angles3 = np.linspace(0, 2 * np.pi, n3, endpoint=False) + phi3
    
    pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
    pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
    pts3 = np.column_stack([r3 * np.cos(angles3), r3 * np.sin(angles3)])
    
    return np.vstack([pts1, pts2, pts3])


def _three_ring_score(params):
    """Score for three-ring parameterization."""
    points = _three_ring_points(params)
    return _score(points)


def _optimize_two_ring(start_params, deadline):
    """Optimize two-ring configuration."""
    try:
        result = minimize(
            lambda p: -_two_ring_score(p),
            start_params,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        if time.time() > deadline:
            return None
        points = _two_ring_points(result.x)
        return points
    except Exception:
        return None


def _optimize_three_ring(start_params, deadline):
    """Optimize three-ring configuration."""
    try:
        result = minimize(
            lambda p: -_three_ring_score(p),
            start_params,
            method='Nelder-Mead',
            options={'maxiter': 800, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        if time.time() > deadline:
            return None
        points = _three_ring_points(result.x)
        return points
    except Exception:
        return None


def _polish_full(points, deadline):
    """Full coordinate polish using L-BFGS-B with soft-min."""
    try:
        result = minimize(
            _soft_min_score,
            points.flatten(),
            args=(200.0,),
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if time.time() > deadline:
            return points
        return result.x.reshape(-1, 2)
    except Exception:
        return points


def construct():
    """
    Construct 14 points to maximize the minimum triangle area divided by convex hull area.
    Uses structural families: two-ring and three-ring, with full-coordinate polish.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin

    best_points = None
    best_score = 0.0

    def update_best(points):
        nonlocal best_points, best_score
        if points is None:
            return
        score = _score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()

    # Two-ring configurations: try different splits
    two_ring_splits = [(7, 7), (8, 6), (9, 5), (10, 4), (11, 3), (12, 2)]
    for n1, n2 in two_ring_splits:
        if time.time() > deadline:
            break
        # Initial params: r1=1.0, phi1=0, r2=0.5, phi2=pi/n1, n1, n2
        init_params = np.array([1.0, 0.0, 0.5, np.pi / n1, n1, n2])
        points = _optimize_two_ring(init_params, deadline)
        update_best(points)
        
        # Try different inner radius
        init_params2 = np.array([1.0, 0.0, 0.7, np.pi / n1, n1, n2])
        points = _optimize_two_ring(init_params2, deadline)
        update_best(points)

    # Three-ring configurations: try different splits
    three_ring_splits = [(7, 4, 3), (8, 3, 3), (9, 3, 2), (10, 2, 2), (6, 4, 4), (5, 5, 4)]
    for n1, n2, n3 in three_ring_splits:
        if time.time() > deadline:
            break
        # Initial params: r1=1.0, phi1=0, r2=0.6, phi2=pi/n1, r3=0.3, phi3=pi/n2
        init_params = np.array([1.0, 0.0, 0.6, np.pi / n1, 0.3, np.pi / n2, n1, n2, n3])
        points = _optimize_three_ring(init_params, deadline)
        update_best(points)
        
        # Try different radii
        init_params2 = np.array([1.0, 0.0, 0.5, np.pi / n1, 0.2, np.pi / n2, n1, n2, n3])
        points = _optimize_three_ring(init_params2, deadline)
        update_best(points)

    # Single ring baseline
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    ring_points = np.column_stack([np.cos(angles), np.sin(angles)])
    update_best(ring_points)

    # Polish the best found
    if best_points is not None and time.time() < deadline - 10.0:
        polished = _polish_full(best_points, deadline)
        update_best(polished)

    if best_points is None:
        # Fallback
        rng = np.random.RandomState(42)
        best_points = rng.uniform(-1, 1, size=(N, 2))

    return best_points
# EVOLVE-BLOCK-END
