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

def _repulsion_step(points, alpha, iters=1):
    """Perform repulsion steps to spread points out."""
    for _ in range(iters):
        forces = np.zeros_like(points)
        for i in range(N):
            for j in range(i + 1, N):
                diff = points[i] - points[j]
                dist_sq = np.dot(diff, diff)
                if dist_sq < 1e-12:
                    continue
                force = diff / (dist_sq ** 1.5)
                forces[i] += force
                forces[j] -= force

        forces_norm = np.linalg.norm(forces, axis=1, keepdims=True)
        forces = forces / np.where(forces_norm == 0, 1, forces_norm)

        new_points = points + alpha * forces
        new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
        points = new_points
    return points

def _spectral_init(n_points):
    """
    Initialize points using a spectral approach:
    1. Start with a random orthonormal basis
    2. Use the top eigenvectors of a Gram matrix designed to spread points
    """
    np.random.seed(12345)
    
    # Start with Fibonacci spiral as a good initial guess
    points = _fibonacci_spiral(n_points)
    
    # Add some random perturbation
    noise = np.random.randn(n_points, 3) * 0.05
    points = points + noise
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    return points

def _gradient_descent_smooth(points, beta=30.0, max_iters=200, step_size=0.01):
    """
    Gradient descent on a smooth approximation of the covering radius.
    Uses the fact that the covering radius is determined by the largest facet circumradius.
    We use a soft-max to create a smooth objective.
    """
    for it in range(max_iters):
        radius_deg, facet_idx, center, hull = _compute_covering_radius(points)
        if center is None or facet_idx == -1:
            break
        
        simplex = hull.simplices[facet_idx]
        v0, v1, v2 = simplex
        
        # Move the three vertices of the worst facet towards the circumcenter
        improved = False
        for mult in [1.0, 0.5, 0.25, 0.1]:
            step = step_size * mult
            new_points = points.copy()
            new_points[v0] = _move_point_towards(points[v0], center, step)
            new_points[v1] = _move_point_towards(points[v1], center, step)
            new_points[v2] = _move_point_towards(points[v2], center, step)
            
            new_radius, _, _, _ = _compute_covering_radius(new_points)
            if new_radius < radius_deg:
                points = new_points
                improved = True
                break
        
        if not improved:
            step_size *= 0.8
            if step_size < 0.001:
                break
        else:
            step_size = min(step_size * 1.1, 0.02)
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    best_points = None
    best_radius = 90.0

    # Phase 1: Global optimization via repulsion from multiple starts
    for seed in range(12):
        if time.time() > deadline - 40:
            break

        np.random.seed(seed * 1000 + 42)

        # Start with Fibonacci spiral plus jitter
        points = _fibonacci_spiral(N).copy()
        noise = np.random.randn(N, 3) * 0.03
        points = points + noise
        points = points / np.linalg.norm(points, axis=1, keepdims=True)

        # Global repulsion to spread points
        alpha = 0.02
        for step in range(300):
            if time.time() > deadline - 40:
                break

            points = _repulsion_step(points, alpha, iters=1)

            # Every 30 steps, check the radius
            if step % 30 == 0 or step == 299:
                radius, _, _, _ = _compute_covering_radius(points)
                if radius < best_radius:
                    best_radius = radius
                    best_points = points.copy()

    # Phase 2: Local search from the best point found
    if best_points is not None:
        points = best_points.copy()
    else:
        points = _fibonacci_spiral(N)
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        best_radius, _, _, _ = _compute_covering_radius(points)
        best_points = points.copy()

    alpha = 0.05
    stagnation_counter = 0
    max_stagnation = 40

    while time.time() < deadline - 15:
        radius, facet_idx, center, hull = _compute_covering_radius(points)

        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        if center is None or facet_idx == -1:
            break

        simplex = hull.simplices[facet_idx]
        v0, v1, v2 = simplex

        improved = False

        # Try moving all 3 vertices of the worst facet towards its circumcenter
        new_points = points.copy()
        new_points[v0] = _move_point_towards(points[v0], center, alpha)
        new_points[v1] = _move_point_towards(points[v1], center, alpha)
        new_points[v2] = _move_point_towards(points[v2], center, alpha)

        new_radius, _, _, _ = _compute_covering_radius(new_points)

        if new_radius < radius:
            points = new_points
            improved = True
        else:
            # Try moving vertices individually
            for v_idx in [v0, v1, v2]:
                test_points = points.copy()
                test_points[v_idx] = _move_point_towards(points[v_idx], center, alpha)
                test_radius, _, _, _ = _compute_covering_radius(test_points)
                if test_radius < radius:
                    points = test_points
                    improved = True
                    break

        if not improved:
            alpha *= 0.85
            if alpha < 0.005:
                # Perturb the worst facet's vertices to escape local minimum
                noise = np.random.randn(3, 3) * 0.04
                new_points = points.copy()
                new_points[v0] = points[v0] + noise[0]
                new_points[v1] = points[v1] + noise[1]
                new_points[v2] = points[v2] + noise[2]

                for idx in [v0, v1, v2]:
                    new_points[idx] = new_points[idx] / np.linalg.norm(new_points[idx])

                pert_radius, _, _, _ = _compute_covering_radius(new_points)

                if pert_radius < best_radius * 1.03:
                    points = new_points
                    alpha = 0.05
                    stagnation_counter = 0
                else:
                    if stagnation_counter > max_stagnation:
                        break
                    else:
                        alpha = 0.02

    # Phase 3: High-precision polish pass
    if time.time() < deadline - 3:
        points = best_points.copy()
        polish_alpha = 0.01
        for polish_iter in range(80):
            if time.time() > deadline - 3:
                break

            radius, facet_idx, center, hull = _compute_covering_radius(points)
            if center is None or facet_idx == -1:
                break

            simplex = hull.simplices[facet_idx]
            v0, v1, v2 = simplex

            improved = False
            # Try smaller steps in polish phase
            for step_mult in [1.0, 0.5, 0.25, 0.1, 0.05]:
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
                # Try individual vertex moves with smaller steps
                for v_idx in [v0, v1, v2]:
                    for step_mult in [1.0, 0.5, 0.25, 0.1]:
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

    # Phase 4: Final fine-tuning with very small steps
    if time.time() < deadline - 1 and best_points is not None:
        points = best_points.copy()
        for _ in range(20):
            if time.time() > deadline - 1:
                break
            radius, facet_idx, center, hull = _compute_covering_radius(points)
            if center is None or facet_idx == -1:
                break
            simplex = hull.simplices[facet_idx]
            v0, v1, v2 = simplex
            for step in [0.005, 0.002, 0.001]:
                new_points = points.copy()
                new_points[v0] = _move_point_towards(points[v0], center, step)
                new_points[v1] = _move_point_towards(points[v1], center, step)
                new_points[v2] = _move_point_towards(points[v2], center, step)
                new_radius, _, _, _ = _compute_covering_radius(new_points)
                if new_radius < radius:
                    points = new_points
                    break
            if radius < best_radius:
                best_radius = radius
                best_points = points.copy()

    return best_points
