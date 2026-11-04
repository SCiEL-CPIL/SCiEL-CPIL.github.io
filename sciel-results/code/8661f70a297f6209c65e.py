"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13
# Precompute all C(13,3) = 286 triangle index triples
import itertools
_TRIPLETS = np.array(list(itertools.combinations(range(N), 3)))
_M = _TRIPLETS.shape[0]  # 286

# Indices for vectorized cross product
_I0 = _TRIPLETS[:, 0]
_I1 = _TRIPLETS[:, 1]
_I2 = _TRIPLETS[:, 2]


def _min_triangle_area(points):
    """Compute minimum triangle area among all C(n,3) triples."""
    p0 = points[_I0]
    p1 = points[_I1]
    p2 = points[_I2]
    # Area = 0.5 * |cross(p1-p0, p2-p0)|
    v1 = p1 - p0
    v2 = p2 - p0
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross).min()


def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    """Score = min triangle area / convex hull area. Returns negative for minimization."""
    points = np.asarray(points, dtype=np.float64)
    ha = _hull_area(points)
    if ha < 1e-12:
        return 0.0
    mta = _min_triangle_area(points)
    return -mta / ha


def _negative_score(points):
    """Objective for minimization: negative of score."""
    return _score(points)


def _regular_polygon(n, scale=1.0):
    """Regular n-gon inscribed in circle of radius scale, centered at origin."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return scale * np.column_stack([np.cos(angles), np.sin(angles)])


def _optimize_from_start(start_points, deadline, max_iter=2000):
    """Run L-BFGS-B from a given starting point. Returns best point found."""
    best_points = start_points.copy()
    best_score = _score(best_points)
    
    start_time = time.time()
    
    def objective(x):
        pts = x.reshape(N, 2)
        return _negative_score(pts)
    
    def callback(xk):
        nonlocal best_points, best_score
        if time.time() > deadline:
            raise ValueError("timeout")
        s = -_score(xk.reshape(N, 2))
        if s > best_score:
            best_score = s
            best_points = xk.reshape(N, 2).copy()
        return False
    
    try:
        res = minimize(
            objective,
            start_points.ravel(),
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'maxfun': 50000, 'ftol': 1e-15, 'gtol': 1e-12},
            callback=callback
        )
        # Check final result
        final_pts = res.x.reshape(N, 2)
        s = -_score(final_pts)
        if s > best_score:
            best_score = s
            best_points = final_pts.copy()
    except Exception:
        pass
    
    return best_points, best_score


def _local_search(points, deadline, n_iter=200, step_size=0.05):
    """Simple coordinate-wise local search / perturbation."""
    best_points = points.copy()
    best_score = _score(best_points)
    rng = np.random.default_rng(42)
    
    for _ in range(n_iter):
        if time.time() > deadline:
            break
        # Try small perturbations
        for attempt in range(20):
            perturbed = best_points + rng.normal(0, step_size, size=(N, 2))
            s = _score(perturbed)
            if s > best_score:
                best_score = s
                best_points = perturbed.copy()
            step_size *= 0.95
        step_size = max(step_size, 0.001)
    
    return best_points, best_score


def construct():
    """Construct 13 points maximizing min triangle area / convex hull area."""
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    
    best_points = None
    best_score = -np.inf
    
    # Strategy 1: Start from regular 13-gon and optimize
    start1 = _regular_polygon(N, scale=1.0)
    pts, sc = _optimize_from_start(start1, deadline, max_iter=3000)
    if sc > best_score:
        best_score = sc
        best_points = pts
    
    if time.time() > deadline:
        return best_points
    
    # Strategy 2: Start from regular 13-gon with slight radial perturbation
    rng = np.random.default_rng(123)
    for trial in range(5):
        if time.time() > deadline:
            break
        # Perturb the regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = 1.0 + rng.normal(0, 0.1, size=N)
        radii = np.clip(radii, 0.5, 1.5)
        start_pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        pts, sc = _optimize_from_start(start_pts, deadline, max_iter=2000)
        if sc > best_score:
            best_score = sc
            best_points = pts
    
    if time.time() > deadline:
        return best_points
    
    # Strategy 3: Start from a different configuration - e.g., two concentric structures
    # Try a configuration with some points on inner ring, some on outer
    for trial in range(3):
        if time.time() > deadline:
            break
        # 7 points on outer ring, 6 on inner ring
        outer_angles = np.linspace(0, 2 * np.pi, 7, endpoint=False) + trial * 0.1
        inner_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False) + trial * 0.15 + 0.3
        outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
        inner_pts = 0.5 * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
        start_pts = np.vstack([outer_pts, inner_pts])
        # Shuffle
        perm = rng.permutation(N)
        start_pts = start_pts[perm]
        pts, sc = _optimize_from_start(start_pts, deadline, max_iter=2000)
        if sc > best_score:
            best_score = sc
            best_points = pts
    
    if time.time() > deadline:
        return best_points
    
    # Strategy 4: Local search polish on best found
    if best_points is not None:
        pts, sc = _local_search(best_points, deadline, n_iter=500, step_size=0.03)
        if sc > best_score:
            best_score = sc
            best_points = pts
    
    if time.time() > deadline:
        return best_points
    
    # Strategy 5: Another round of L-BFGS from the polished point with tighter tolerance
    if best_points is not None:
        pts, sc = _optimize_from_start(best_points, deadline, max_iter=5000)
        if sc > best_score:
            best_score = sc
            best_points = pts
    
    return best_points
