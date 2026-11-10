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

def _smooth_covering_radius(points, beta=50.0):
    """
    Compute a smooth approximation of the covering radius using log-sum-exp.
    Returns (smooth_radius_radians, gradient_wrt_points).
    The gradient has shape (N, 3) and points in the direction that DECREASES the radius.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return 90.0 * math.pi / 180.0, np.zeros_like(points)

    # Compute circumradius for each facet
    radii = []
    centers = []
    valid_indices = []
    
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
        radii.append(R)
        centers.append(C)
        valid_indices.append(i)

    if not radii:
        return 90.0 * math.pi / 180.0, np.zeros_like(points)

    radii = np.array(radii)
    
    # Log-sum-exp for smooth max
    max_r = np.max(radii)
    if max_r < 1e-10:
        return 0.0, np.zeros_like(points)
    
    # Smooth max: (1/beta) * log(sum(exp(beta * r_i)))
    shifted = beta * (radii - max_r)
    log_sum_exp = max_r + (1.0 / beta) * np.log(np.sum(np.exp(shifted)))
    
    # Weights for each facet (softmax)
    weights = np.exp(shifted)
    weights = weights / np.sum(weights)
    
    # Gradient: d(log_sum_exp)/d(points) = sum_i weights_i * d(r_i)/d(points)
    # d(r_i)/d(p_j) where p_j is a vertex of facet i
    # r_i = acos(1/sqrt(x^T M x))
    # This is complex; use finite differences or analytical gradient
    
    # For now, use a simpler approach: gradient points away from circumcenter
    gradient = np.zeros_like(points)
    
    for idx, (i, R) in enumerate(zip(valid_indices, radii)):
        simplex = hull.simplices[i]
        C = centers[idx]
        w_i = weights[idx]
        
        # The circumcenter C is the point on the sphere farthest from the facet
        # Moving vertices towards C should decrease the circumradius
        for v_idx in simplex:
            p = points[v_idx]
            # Tangent direction from p towards C
            dot_pc = np.dot(p, C)
            t = C - dot_pc * p
            norm_t = np.linalg.norm(t)
            if norm_t > 1e-10:
                t = t / norm_t
                # Moving towards C decreases radius, so gradient is -t
                gradient[v_idx] += w_i * R * t
    
    # Normalize gradient
    grad_norm = np.linalg.norm(gradient, axis=1, keepdims=True)
    gradient = gradient / np.where(grad_norm == 0, 1, grad_norm)
    
    return log_sum_exp, gradient

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    np.random.seed(42)

    best_points = None
    best_radius = 90.0

    # Phase 1: Global optimization via repulsion from multiple starts
    for seed in range(8):
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

    # Phase 2: Local search using smooth gradient
    if best_points is not None:
        points = best_points.copy()
    else:
        points = _fibonacci_spiral(N)
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        best_radius, _, _, _ = _compute_covering_radius(points)
        best_points = points.copy()

    # Use smooth gradient-based optimization
    alpha = 0.03
    beta = 30.0  # Start with moderate smoothing
    stagnation_counter = 0
    max_stagnation = 60

    while time.time() < deadline - 15:
        # Compute smooth radius and gradient
        smooth_rad, gradient = _smooth_covering_radius(points, beta)
        
        # Also compute exact radius for tracking
        exact_radius, facet_idx, center, hull = _compute_covering_radius(points)
        
        if exact_radius < best_radius:
            best_radius = exact_radius
            best_points = points.copy()
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        # Gradient descent step: move points in direction that decreases radius
        # gradient points in direction of decrease
        new_points = points + alpha * gradient
        new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
        
        new_radius, _, _, _ = _compute_covering_radius(new_points)
        
        if new_radius < exact_radius:
            points = new_points
            # Slowly decrease alpha and increase beta for more precise optimization
            alpha *= 0.98
            beta = min(beta * 1.02, 100.0)
        else:
            # Try smaller step
            for step_mult in [0.5, 0.25, 0.1]:
                test_points = points + alpha * step_mult * gradient
                test_points = test_points / np.linalg.norm(test_points, axis=1, keepdims=True)
                test_radius, _, _, _ = _compute_covering_radius(test_points)
                if test_radius < exact_radius:
                    points = test_points
                    alpha *= 0.95
                    break
            else:
                # No improvement, adjust parameters
                alpha *= 0.8
                if alpha < 0.005:
                    # Perturb to escape local minimum
                    noise = np.random.randn(N, 3) * 0.02
                    pert_points = points + noise
                    pert_points = pert_points / np.linalg.norm(pert_points, axis=1, keepdims=True)
                    pert_radius, _, _, _ = _compute_covering_radius(pert_points)
                    
                    if pert_radius < best_radius * 1.05:
                        points = pert_points
                        alpha = 0.03
                        beta = 30.0
                        stagnation_counter = 0
                    else:
                        if stagnation_counter > max_stagnation:
                            break
                        else:
                            alpha = 0.02
                            beta = max(beta * 0.9, 20.0)

    # Phase 3: High-precision polish pass with discrete facet-based moves
    if time.time() < deadline - 3:
        points = best_points.copy()
        polish_alpha = 0.01
        for polish_iter in range(100):
            if time.time() > deadline - 3:
                break

            radius, facet_idx, center, hull = _compute_covering_radius(points)
            if center is None or facet_idx == -1:
                break

            simplex = hull.simplices[facet_idx]
            v0, v1, v2 = simplex

            improved = False
            # Try smaller steps in polish phase
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
                # Try individual vertex moves with smaller steps
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
