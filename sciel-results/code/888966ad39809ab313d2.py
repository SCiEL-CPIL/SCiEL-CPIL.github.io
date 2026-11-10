import numpy as np
import time
from scipy.spatial import ConvexHull
import math

N = 130

def _compute_covering_radius(points):
    """
    Computes the covering radius (max circumradius of hull facets) in degrees.
    Returns (radius_in_degrees, worst_facet_index, worst_circumcenter, hull).
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return 90.0, -1, None, None

    max_radius = 0.0
    worst_facet_idx = -1
    worst_center = None

    for i, simplex in enumerate(hull.simplices):
        u = points[simplex[0]]
        v = points[simplex[1]]
        w = points[simplex[2]]

        uv = np.dot(u, v)
        uw = np.dot(u, w)
        vw = np.dot(v, w)

        M = np.array([
            [1.0, uv, uw],
            [uv, 1.0, vw],
            [uw, vw, 1.0]
        ])

        try:
            x = np.linalg.solve(M, np.ones(3))
        except np.linalg.LinAlgError:
            continue

        xT_M_x = np.dot(x, M @ x)

        if xT_M_x <= 1e-10:
            continue

        cos_R = 1.0 / np.sqrt(xT_M_x)
        cos_R = np.clip(cos_R, -1.0, 1.0)

        C = x[0] * u + x[1] * v + x[2] * w
        norm_C = np.linalg.norm(C)
        if norm_C < 1e-10:
            continue
        C = C / norm_C

        R = math.acos(cos_R)

        if R > max_radius:
            max_radius = R
            worst_center = C
            worst_facet_idx = i

    return math.degrees(max_radius), worst_facet_idx, worst_center, hull

def _move_point_towards(p, c, step_size):
    """Move point p on the sphere towards c by step_size radians."""
    dot_pc = np.dot(p, c)
    t = c - dot_pc * p
    norm_t = np.linalg.norm(t)
    if norm_t < 1e-10:
        return p
    t = t / norm_t
    new_p = p + step_size * t
    new_p = new_p / np.linalg.norm(new_p)
    return new_p

def _fibonacci_spiral(n):
    """Generate a Fibonacci spiral configuration on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return points

def _repulsion_step_vectorized(points, alpha):
    """Perform vectorized repulsion step to spread points out."""
    # Compute pairwise differences
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    
    # Avoid division by zero
    dist_sq[dist_sq < 1e-12] = 1e-12
    
    # Force: diff / dist^1.5
    forces = diff / (dist_sq ** 1.5)[:, :, np.newaxis]
    
    # Sum forces for each point (excluding self)
    np.fill_diagonal(forces, 0.0)
    total_forces = np.sum(forces, axis=1)
    
    # Normalize forces
    forces_norm = np.linalg.norm(total_forces, axis=1, keepdims=True)
    forces_norm[forces_norm == 0] = 1.0
    total_forces = total_forces / forces_norm
    
    new_points = points + alpha * total_forces
    new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
    return new_points

def _spectral_init(n):
    """Initialize using spectral method - distribute points evenly on sphere."""
    # Start with Fibonacci spiral as base
    points = _fibonacci_spiral(n).copy()
    
    # Apply a few repulsion steps to spread out
    for i in range(50):
        alpha = 0.01 * (1 - i / 50.0) + 0.001
        points = _repulsion_step_vectorized(points, alpha)
    
    return points

def _lloyd_iteration(points, max_iters=50, time_budget=30.0):
    """
    Lloyd's iteration: assign each point to its nearest facet center,
    then move towards that center. This is like k-means on the sphere.
    """
    start_time = time.time()
    points = points.copy()
    
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        radius, facet_idx, center, hull = _compute_covering_radius(points)
        if center is None or facet_idx == -1:
            break
        
        # For each point, find which facet it's closest to (by circumcenter)
        # and move it towards that facet's circumcenter
        # Simplified: move all points towards the worst facet center
        simplex = hull.simplices[facet_idx]
        v0, v1, v2 = simplex
        
        # Move the three vertices of the worst facet towards its center
        step = 0.05
        new_points = points.copy()
        new_points[v0] = _move_point_towards(points[v0], center, step)
        new_points[v1] = _move_point_towards(points[v1], center, step)
        new_points[v2] = _move_point_towards(points[v2], center, step)
        
        new_radius, _, _, _ = _compute_covering_radius(new_points)
        if new_radius < radius:
            points = new_points
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin

    np.random.seed(42)

    best_points = None
    best_radius = 90.0

    # Phase 1: Global optimization via repulsion from multiple starts
    for seed in range(6):
        if time.time() > deadline - 45:
            break

        np.random.seed(seed * 1000 + 42)

        # Start with spectral initialization plus jitter
        points = _spectral_init(N).copy()
        noise = np.random.randn(N, 3) * 0.02
        points = points + noise
        points = points / np.linalg.norm(points, axis=1, keepdims=True)

        # Global repulsion to spread points using vectorized version
        alpha = 0.01
        for step in range(200):
            if time.time() > deadline - 45:
                break

            points = _repulsion_step_vectorized(points, alpha)
            alpha *= 0.995

            # Every 20 steps, check the radius
            if step % 20 == 0 or step == 199:
                radius, _, _, _ = _compute_covering_radius(points)
                if radius < best_radius:
                    best_radius = radius
                    best_points = points.copy()

    # Phase 2: Lloyd's iteration for better local optimization
    if best_points is not None:
        points = best_points.copy()
    else:
        points = _spectral_init(N)
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        best_radius, _, _, _ = _compute_covering_radius(points)
        best_points = points.copy()

    # Apply Lloyd's iteration
    if time.time() < deadline - 25:
        points = _lloyd_iteration(points, max_iters=100, time_budget=20.0)
        radius, _, _, _ = _compute_covering_radius(points)
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()

    # Phase 3: High-precision polish pass
    if time.time() < deadline - 5:
        points = best_points.copy()
        polish_alpha = 0.02
        for polish_iter in range(80):
            if time.time() > deadline - 5:
                break

            radius, facet_idx, center, hull = _compute_covering_radius(points)
            if center is None or facet_idx == -1:
                break

            simplex = hull.simplices[facet_idx]
            v0, v1, v2 = simplex

            improved = False
            for step_mult in [1.0, 0.5, 0.25, 0.1, 0.05, 0.02]:
                step = polish_alpha * step_mult
                new_points = points.copy()
                new_points[v0] = _move_point_towards(points[v0], center, step)
                new_points[v1] = _move_point_towards(points[v1], center, step)
                new_points[v2] = _move_point_towards(points[v2], center, step)

                new_radius, _, _, _ = _compute_covering_radius(new_points)
                if new_radius < radius:
                    points = new_points
                    improved = True
                    break

            if not improved:
                # Try individual vertex moves
                for v_idx in [v0, v1, v2]:
                    for step_mult in [1.0, 0.5, 0.25, 0.1, 0.05]:
                        step = polish_alpha * step_mult
                        test_points = points.copy()
                        test_points[v_idx] = _move_point_towards(points[v_idx], center, step)
                        test_radius, _, _, _ = _compute_covering_radius(test_points)
                        if test_radius < radius:
                            points = test_points
                            improved = True
                            break
                    if improved:
                        break

            if not improved:
                polish_alpha *= 0.7
                if polish_alpha < 0.0005:
                    break
            else:
                if radius < best_radius:
                    best_radius = radius
                    best_points = points.copy()

    return best_points
