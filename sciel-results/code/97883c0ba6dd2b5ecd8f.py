"""Rewritten program for Heilbronn convex regions n=14."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize


def _min_triangle_area(pts):
    """Compute the minimum triangle area over all triples of points."""
    n = len(pts)
    min_area = np.inf
    # Vectorized over pairs for speed
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            v = pts[j] - pts[i]
            # Cross product magnitude with all points k > j
            rest = pts[j + 1:]
            # area = 0.5 * |cross(v, rest - pts[i])|
            cross = np.abs(v[0] * (rest[:, 1] - pts[i, 1]) - v[1] * (rest[:, 0] - pts[i, 0]))
            if cross.size > 0:
                m = cross.min()
                if m < min_area:
                    min_area = m
    return 0.5 * min_area


def _convex_hull_area(pts):
    """Compute the area of the convex hull of points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 1.0


def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 1e-12:
        return 0.0
    return _min_triangle_area(pts) / hull_area


def _regular_polygon(n, radius=1.0):
    """Generate a regular n-gon centered at origin."""
    angles = 2 * np.pi * np.arange(n) / n
    return radius * np.column_stack([np.cos(angles), np.sin(angles)])


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    # Start with a regular polygon (symmetric, good initial guess)
    pts = _regular_polygon(N, radius=1.0)
    best_pts = pts.copy()
    best_score = _score(pts)

    # Use Nelder-Mead to maximize the score (minimize negative score)
    # Multiple restarts with perturbations
    rng = np.random.default_rng(42)

    def objective(x):
        p = x.reshape(N, 2)
        s = _score(p)
        return -s

    # First, optimize from the regular polygon
    try:
        res = minimize(objective, pts.flatten(), method='Nelder-Mead',
                       options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10,
                                'adaptive': True})
        if -res.fun > best_score:
            best_score = -res.fun
            best_pts = res.x.reshape(N, 2)
    except Exception:
        pass

    # Check time
    if time.time() > deadline - 30:
        return best_pts

    # Try perturbations and re-optimize
    for restart in range(10):
        if time.time() > deadline - 10:
            break

        # Perturb the best solution
        perturbation = rng.normal(0, 0.1, size=(N, 2))
        x0 = best_pts + perturbation

        try:
            res = minimize(objective, x0.flatten(), method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10,
                                    'adaptive': True})
            if -res.fun > best_score:
                best_score = -res.fun
                best_pts = res.x.reshape(N, 2)
        except Exception:
            pass

    # Also try a different initial configuration: points on a circle with some interior points
    if time.time() < deadline - 20:
        # 8 points on outer circle, 6 points in inner region
        outer_angles = 2 * np.pi * np.arange(8) / 8
        outer = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
        inner_angles = 2 * np.pi * np.arange(6) / 6 + np.pi / 6
        inner = 0.5 * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
        pts2 = np.vstack([outer, inner])
        best_pts2 = pts2.copy()
        best_score2 = _score(pts2)

        try:
            res = minimize(objective, pts2.flatten(), method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10,
                                    'adaptive': True})
            if -res.fun > best_score2:
                best_score2 = -res.fun
                best_pts2 = res.x.reshape(N, 2)
        except Exception:
            pass

        if best_score2 > best_score:
            best_score = best_score2
            best_pts = best_pts2

    return best_pts
# EVOLVE-BLOCK-END
