"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import ConvexHull, Delaunay
from itertools import combinations

N = 13

# EVOLVE-BLOCK-START
import random
from scipy.optimize import minimize


def _min_triangle_area(pts):
    """Compute the minimum area over all triples of points."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(pts):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0


def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 0.0
    min_area = _min_triangle_area(pts)
    return min_area / hull_area


def _soft_min_triangle_area(pts, beta=100.0):
    """Soft minimum of triangle areas for differentiability."""
    n = len(pts)
    areas = np.zeros((n, n, n))
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                areas[i, j, k] = area
                areas[j, i, k] = area
                areas[j, k, i] = area
                areas[k, i, j] = area
                areas[k, j, i] = area
                areas[i, k, j] = area
    # Soft min using log-sum-exp trick
    areas_flat = areas.flatten()
    valid = areas_flat > 1e-12
    if not np.any(valid):
        return 0.0
    areas_valid = areas_flat[valid]
    # Use a soft minimum approximation
    soft_min = np.exp(-beta * np.logsumexp(-beta * areas_valid))
    return soft_min


def _optimize_from_initial(pts0, time_limit=140.0):
    """Optimize points starting from pts0, with a time limit."""
    start_time = time.time()
    best_score = _score(pts0)
    best_pts = pts0.copy()
    
    # Use L-BFGS-B with a differentiable objective
    # We maximize min_area / hull_area
    # Equivalently, minimize -min_area / hull_area
    
    def objective(x):
        if time.time() - start_time > time_limit - 5:
            return 1e10  # Large penalty if we're running out of time
        pts = x.reshape(N, 2)
        hull_area = _convex_hull_area(pts)
        if hull_area <= 1e-12:
            return 1e10
        min_area = _min_triangle_area(pts)
        return -min_area / hull_area
    
    def objective_grad(x):
        # Numerical gradient
        eps = 1e-7
        grad = np.zeros_like(x)
        f0 = objective(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            f_plus = objective(x_plus)
            grad[i] = (f_plus - f0) / eps
        return grad
    
    x0 = pts0.flatten()
    
    # Try multiple beta values for soft min approach
    # First, try direct optimization with numerical gradient
    result = minimize(objective, x0, method='L-BFGS-B', 
                      options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
    
    if result.success or result.fun < 1e9:
        pts_opt = result.x.reshape(N, 2)
        score_opt = _score(pts_opt)
        if score_opt > best_score:
            best_score = score_opt
            best_pts = pts_opt.copy()
    
    # Try a second round with perturbed initial points
    if time.time() - start_time < time_limit - 10:
        rng = np.random.RandomState(42)
        for trial in range(3):
            if time.time() - start_time > time_limit - 10:
                break
            # Perturb the best points found so far
            perturbation_scale = 0.1 / (trial + 1)
            x_pert = best_pts.flatten() + rng.randn(2 * N) * perturbation_scale
            result2 = minimize(objective, x_pert, method='L-BFGS-B',
                               options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            if result2.fun < 1e9:
                pts_opt2 = result2.x.reshape(N, 2)
                score_opt2 = _score(pts_opt2)
                if score_opt2 > best_score:
                    best_score = score_opt2
                    best_pts = pts_opt2.copy()
    
    return best_pts, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Initial guess: Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    best_pts, best_score = _optimize_from_initial(points, time_limit=deadline - start_time)
    
    # If we have time, try another initial configuration
    if time.time() < deadline - 20:
        # Try a slightly different initial configuration
        angles2 = np.linspace(0, 2 * np.pi, N, endpoint=False) + 0.1
        points2 = np.column_stack([np.cos(angles2), np.sin(angles2)])
        pts2, score2 = _optimize_from_initial(points2, time_limit=deadline - time.time())
        if score2 > best_score:
            best_pts, best_score = pts2, score2
    
    # Ensure we're within the convex region and return
    return best_pts
# EVOLVE-BLOCK-END
