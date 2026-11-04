"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
import itertools

N = 13

# Precompute all triangle indices (i, j, k) with i < j < k
_TRIPLETS = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)])
_I, _J, _K = _TRIPLETS[:, 0], _TRIPLETS[:, 1], _TRIPLETS[:, 2]


def _triangle_areas(pts):
    """Vectorized computation of all triangle areas from N points."""
    # pts shape (N, 2)
    pi = pts[_I]  # (T, 2)
    pj = pts[_J]
    pk = pts[_K]
    # Area = 0.5 * |cross(pj - pi, pk - pi)|
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)


def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _score(pts):
    """Return min triangle area / convex hull area."""
    pts = np.asarray(pts)
    hull_a = _hull_area(pts)
    if hull_a < 1e-12:
        return 0.0
    areas = _triangle_areas(pts)
    min_a = np.min(areas)
    return min_a / hull_a


def _neg_score(pts):
    """Negative score for minimization."""
    return -_score(pts)


def _soft_min(areas, beta=100.0):
    """Soft minimum via log-sum-exp (smooth surrogate for min)."""
    # -log(sum(exp(-beta * areas))) / beta approximates min(areas)
    # Use stable computation
    m = np.min(areas)
    return m - np.log(np.sum(np.exp(-beta * (areas - m)))) / beta


def _neg_soft_score(pts, beta=50.0):
    """Smooth objective: -soft_min(areas) / hull_area."""
    pts = np.asarray(pts)
    hull_a = _hull_area(pts)
    if hull_a < 1e-12:
        return 0.0
    areas = _triangle_areas(pts)
    sm = _soft_min(areas, beta)
    return -sm / hull_a


def _local_polish(pts, max_iter=200, tol=1e-10):
    """L-BFGS-B polish in Cartesian space."""
    x0 = pts.flatten()
    res = minimize(
        _neg_score,
        x0,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8}
    )
    best_pts = res.x.reshape(N, 2)
    return best_pts, _score(best_pts)


def _soft_polish(pts, beta=50.0, max_iter=100, tol=1e-8):
    """Polish using soft-min surrogate for smoother gradients."""
    x0 = pts.flatten()
    res = minimize(
        lambda x: _neg_soft_score(x.reshape(N, 2), beta),
        x0,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-6}
    )
    best_pts = res.x.reshape(N, 2)
    return best_pts, _score(best_pts)


def _random_start(rng):
    """Generate random points in unit square."""
    return np.array([[rng.random(), rng.random()] for _ in range(N)])


def _circle_start(r, n=N):
    """Points on a circle of radius r."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([r * np.cos(angles), r * np.sin(angles)])


def _grid_start():
    """A structured grid-like start."""
    # 13 points: roughly a 4x4 grid with one removed, scaled
    pts = []
    for i in range(4):
        for j in range(4):
            if i == 3 and j == 3:
                continue
            pts.append([i / 3.0, j / 3.0])
    pts.append([0.5, 0.5])  # center
    return np.array(pts[:N])


def construct():
    start_time = time.time()
    deadline = 120.0  # safety margin before 150s kill

    rng = np.random.RandomState(42)

    best_pts = None
    best_score = 0.0

    # Initial diverse starts
    starts = []
    # Random starts
    for seed in range(20):
        r = np.random.RandomState(seed)
        starts.append(np.array([[r.random(), r.random()] for _ in range(N)]))
    # Circle starts with varying radii
    for r in [0.3, 0.5, 0.7, 1.0]:
        starts.append(_circle_start(r))
    # Grid start
    starts.append(_grid_start())

    # Evaluate and polish each start
    for idx, start_pts in enumerate(starts):
        if time.time() > deadline - 5.0:
            break

        # Quick score
        s = _score(start_pts)
        if s > best_score:
            best_score = s
            best_pts = start_pts.copy()

        # Soft polish (smoother landscape)
        polished, s_pol = _soft_polish(start_pts, beta=50.0, max_iter=80)
        if s_pol > best_score:
            best_score = s_pol
            best_pts = polished.copy()

        # Hard polish (exact min)
        polished2, s_pol2 = _local_polish(polished, max_iter=150, tol=1e-10)
        if s_pol2 > best_score:
            best_score = s_pol2
            best_pts = polished2.copy()

        # Check time
        if time.time() > deadline - 10.0:
            break

    # If we have time, do additional random restarts with polish
    while time.time() < deadline - 10.0:
        r = np.random.RandomState()
        new_start = np.array([[r.random(), r.random()] for _ in range(N)])
        s = _score(new_start)
        if s > best_score:
            best_score = s
            best_pts = new_start.copy()

        # Quick soft polish
        polished, s_pol = _soft_polish(new_start, beta=50.0, max_iter=50)
        if s_pol > best_score:
            best_score = s_pol
            best_pts = polished.copy()

        # Hard polish
        polished2, s_pol2 = _local_polish(polished, max_iter=100, tol=1e-10)
        if s_pol2 > best_score:
            best_score = s_pol2
            best_pts = polished2.copy()

    # Final high-precision polish on best
    if best_pts is not None and time.time() < deadline - 5.0:
        final_pts, final_s = _local_polish(best_pts, max_iter=300, tol=1e-12)
        if final_s > best_score:
            best_score = final_s
            best_pts = final_pts

    if best_pts is None:
        # Fallback
        return np.array([[rng.random(), rng.random()] for _ in range(N)])

    return best_pts
