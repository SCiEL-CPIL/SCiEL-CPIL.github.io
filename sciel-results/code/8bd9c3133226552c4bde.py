"""Heilbronn convex regions: place 13 points to maximize minimum triangle area / hull area."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13
TIME_LIMIT = 110.0  # seconds, safety margin before 120s kill


def _compute_min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 of the given points."""
    n = points.shape[0]
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    return min_area


def _compute_hull_area(points):
    """Compute the area of the convex hull of the given points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    min_tri = _compute_min_triangle_area(points)
    return min_tri / hull_area


def _objective(x):
    """Objective to minimize: negative score (we want to maximize score)."""
    points = x.reshape(-1, 2)
    s = _score(points)
    return -s


def _regular_polygon(n):
    """Return vertices of a regular n-gon centered at origin with radius 1."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def _optimize_from_start(start_points, deadline):
    """Run L-BFGS-B optimization from a given starting configuration."""
    x0 = start_points.flatten()
    
    def obj(x):
        if time.time() > deadline:
            raise TimeoutError()
        return _objective(x)
    
    try:
        result = minimize(
            obj,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 200, 'maxfun': 5000}
        )
        if time.time() > deadline:
            return start_points, _score(start_points)
        best_points = result.x.reshape(-1, 2)
        best_score = _score(best_points)
        return best_points, best_score
    except (TimeoutError, Exception):
        return start_points, _score(start_points)


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_points = None
    best_score = -float('inf')
    
    # Strategy: use regular 13-gon as base, perturb radially and angularly
    # Try multiple random perturbations of the regular polygon
    
    base = _regular_polygon(N)
    
    # Initial: try the regular polygon itself
    s = _score(base)
    if s > best_score:
        best_score = s
        best_points = base.copy()
    
    # Try random perturbations
    rng = np.random.RandomState(42)
    
    num_tries = 0
    max_tries = 5000
    
    while time.time() < deadline - 2.0 and num_tries < max_tries:
        num_tries += 1
        
        # Random perturbation: radial and angular noise
        noise_scale = 0.3
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = 1.0 + noise_scale * rng.randn(N)
        angle_offsets = noise_scale * 0.5 * rng.randn(N)
        
        pts = np.column_stack([
            radii * np.cos(angles + angle_offsets),
            radii * np.sin(angles + angle_offsets)
        ])
        
        # Run a quick optimization from this start
        opt_pts, opt_score = _optimize_from_start(pts, deadline)
        
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_pts.copy()
        
        # Also try small perturbations of the current best
        if num_tries % 5 == 0 and best_points is not None:
            small_noise = 0.05 * rng.randn(N, 2)
            perturbed = best_points + small_noise
            opt_pts2, opt_score2 = _optimize_from_start(perturbed, deadline)
            if opt_score2 > best_score:
                best_score = opt_score2
                best_points = opt_pts2.copy()
    
    # Final: one more optimization from the best found
    if best_points is not None and time.time() < deadline - 1.0:
        final_pts, final_score = _optimize_from_start(best_points, deadline)
        if final_score > best_score:
            best_score = final_score
            best_points = final_pts.copy()
    
    if best_points is None:
        best_points = base
    
    return best_points
