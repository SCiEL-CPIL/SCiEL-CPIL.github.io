"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
import itertools
import time


def _min_triangle_area(points):
    """Compute the minimum triangle area among all triples of points using vectorized operations."""
    n = len(points)
    # Precompute all pairwise differences
    # Use broadcasting to compute areas for all triples efficiently
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            v1 = points[j] - points[i]
            # Compute areas for all k > j
            v2 = points[j+1:] - points[i]
            areas = 0.5 * np.abs(v1[0] * v2[:, 1] - v1[1] * v2[:, 0])
            if len(areas) > 0:
                local_min = np.min(areas)
                if local_min < min_area:
                    min_area = local_min
    return min_area


def _convex_hull_area(points):
    """Compute the area of the convex hull using Andrew's monotone chain."""
    pts = np.array(points)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    # Build lower hull
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    
    # Build upper hull
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    
    # Concatenate
    hull = lower[:-1] + upper[:-1]
    hull = np.array(hull)
    
    if len(hull) < 3:
        return 0.0
    
    # Shoelace formula
    x = hull[:, 0]
    y = hull[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return area


def _is_strictly_convex(points):
    """Check if points form a strictly convex polygon (in given order)."""
    n = len(points)
    if n < 3:
        return False
    for i in range(n):
        p0 = points[i]
        p1 = points[(i + 1) % n]
        p2 = points[(i + 2) % n]
        cross = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
        if cross <= 1e-10:
            return False
    return True


def _score_from_params(params):
    """
    Convert perturbation parameters to points and compute score.
    params: array of shape (2*N,)
      - params[0:N] are angular perturbations
      - params[N:2*N] are radial perturbations
    """
    base_angles = 2 * np.pi * np.arange(N) / N
    base_radius = 1.0
    
    angles = base_angles + params[0:N]
    radii = base_radius + params[N:2*N]
    
    # Ensure radii are positive
    radii = np.maximum(radii, 0.1)
    
    points = np.column_stack([
        radii * np.cos(angles),
        radii * np.sin(angles)
    ])
    
    # Sort by angle to ensure convex order
    sorted_idx = np.argsort(angles)
    points = points[sorted_idx]
    
    if not _is_strictly_convex(points):
        return 0.0
    
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    
    if hull_area <= 1e-10:
        return 0.0
    
    return min_area / hull_area


def _regular_n_gon(n):
    """Generate a regular n-gon centered at origin."""
    angles = 2 * np.pi * np.arange(n) / n
    return np.column_stack([np.cos(angles), np.sin(angles)])


def _local_optimize(points, max_iter=100, step_size=0.01):
    """Perform local optimization using coordinate descent on angular and radial params."""
    # Extract current angles and radii
    angles = np.arctan2(points[:, 1], points[:, 0])
    radii = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    
    # Sort by angle
    sorted_idx = np.argsort(angles)
    angles = angles[sorted_idx]
    radii = radii[sorted_idx]
    
    best_score = _score_from_params(np.concatenate([angles - 2*np.pi*np.arange(N)/N, radii - 1.0]))
    best_params = np.concatenate([angles - 2*np.pi*np.arange(N)/N, radii - 1.0]).copy()
    
    for iteration in range(max_iter):
        improved = False
        # Try perturbing each angular parameter
        for i in range(N):
            for delta in [-step_size, step_size]:
                test_params = best_params.copy()
                test_params[i] += delta
                score = _score_from_params(test_params)
                if score > best_score + 1e-10:
                    best_params = test_params
                    best_score = score
                    improved = True
        # Try perturbing each radial parameter
        for i in range(N):
            for delta in [-step_size, step_size]:
                test_params = best_params.copy()
                test_params[N + i] += delta
                score = _score_from_params(test_params)
                if score > best_score + 1e-10:
                    best_params = test_params
                    best_score = score
                    improved = True
        if not improved:
            break
    
    return best_params, best_score


def construct():
    """
    Construct 14 points in a planar convex region normalized to unit area
    to maximize the minimum triangle area.
    
    Returns:
        np.array of shape (14, 2)
    """
    start_time = time.time()
    time_limit = 110  # Leave some buffer for the 120s limit
    
    # Start from regular 14-gon
    base_angles = 2 * np.pi * np.arange(N) / N
    initial_points = _regular_n_gon(N)
    
    # Initial score
    initial_score = _score_from_params(np.zeros(2*N))
    
    # Stage 1: Differential evolution with moderate settings
    max_angle_pert = 0.3 * (2 * np.pi / N)
    max_radius_pert = 0.4
    
    bounds = [(-max_angle_pert, max_angle_pert)] * N + [(-max_radius_pert, max_radius_pert)] * N
    
    def objective(params):
        return -_score_from_params(params)
    
    # Run differential evolution
    best_params = np.zeros(2*N)
    best_score = initial_score
    
    if time.time() - start_time < time_limit * 0.6:
        try:
            result = differential_evolution(
                objective,
                bounds,
                maxiter=30,
                popsize=12,
                tol=1e-5,
                seed=42,
                polish=False
            )
            de_score = -result.fun
            if de_score > best_score:
                best_params = result.x
                best_score = de_score
        except Exception:
            pass
    
    # Stage 2: Local optimization
    if time.time() - start_time < time_limit * 0.9:
        try:
            local_params, local_score = _local_optimize(
                np.column_stack([
                    (1.0 + best_params[N:2*N]) * np.cos(base_angles + best_params[0:N]),
                    (1.0 + best_params[N:2*N]) * np.sin(base_angles + best_params[0:N])
                ]),
                max_iter=50,
                step_size=0.02
            )
            if local_score > best_score:
                best_params = local_params
                best_score = local_score
        except Exception:
            pass
    
    # Convert best params to points
    angles = base_angles + best_params[0:N]
    radii = 1.0 + best_params[N:2*N]
    radii = np.maximum(radii, 0.1)
    
    points = np.column_stack([
        radii * np.cos(angles),
        radii * np.sin(angles)
    ])
    
    # Sort by angle
    sorted_idx = np.argsort(angles)
    points = points[sorted_idx]
    
    # Normalize to unit area by scaling
    hull_area = _convex_hull_area(points)
    if hull_area > 1e-10:
        scale = 1.0 / np.sqrt(hull_area)
        points = points * scale
    
    return points
# EVOLVE-BLOCK-END
