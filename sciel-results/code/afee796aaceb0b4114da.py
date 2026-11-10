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
    """Perform vectorized repulsion steps to spread points out."""
    for _ in range(iters):
        # Compute all pairwise differences: (N, N, 3)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)  # (N, N)
        
        # Avoid division by zero on diagonal
        np.fill_diagonal(dist_sq, 1.0)
        
        # Force magnitude: 1 / dist^3 (since diff/dist gives direction, and 1/dist^2 gives magnitude)
        # force_ij = diff_ij / dist_sq_ij^(3/2)
        force_mag = 1.0 / (dist_sq ** 1.5)
        np.fill_diagonal(force_mag, 0.0)
        
        # forces[i] = sum_j force_mag[i,j] * diff[i,j]
        forces = np.einsum('ij,ijk->ik', force_mag, diff)
        
        # Normalize forces to unit length for stable stepping
        forces_norm = np.linalg.norm(forces, axis=1, keepdims=True)
        forces = forces / np.where(forces_norm < 1e-10, 1.0, forces_norm)
        
        new_points = points + alpha * forces
        new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
        points = new_points
    return points

def _spectral_init(n):
    """
    Generate initial points using a spectral approach:
    1. Start with Fibonacci spiral
    2. Use eigendecomposition of a distance-based matrix to spread points
    """
    points = _fibonacci_spiral(n).copy()
    
    alpha_kernel = 5.0
    dist_sq = np.sum(points**2, axis=1)[:, None] + np.sum(points**2, axis=1)[None, :] - 2 * points @ points.T
    dist_sq = np.maximum(dist_sq, 0)
    K = np.exp(-alpha_kernel * dist_sq)
    K += np.eye(n) * 0.01
    
    try:
        eigvals, eigvecs = np.linalg.eigh(K)
        idx = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, idx]
        
        for k in range(min(10, n)):
            weights = np.abs(eigvecs[:, k])
            weights = weights / (np.sum(weights) + 1e-10)
            points = 0.7 * points + 0.3 * (weights[:, None] * points)
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
    except Exception:
        pass
    
    return points

def _smooth_covering_radius_and_gradient(points, beta=50.0):
    """
    Compute smooth log-sum-exp approximation of covering radius and its gradient.
    Returns (smooth_radius_radians, gradient_array_of_shape_Nx3).
    Gradient points in direction that DECREASES the smooth radius.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return 90.0 * math.pi / 180.0, np.zeros_like(points)

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
    max_r = np.max(radii)
    if max_r < 1e-10:
        return 0.0, np.zeros_like(points)
    
    # Log-sum-exp
    shifted = beta * (radii - max_r)
    log_sum_exp = max_r + (1.0 / beta) * np.log(np.sum(np.exp(shifted)))
    
    # Softmax weights
    weights = np.exp(shifted)
    weights = weights / np.sum(weights)
    
    # Gradient: for each facet, moving vertices towards circumcenter decreases radius
    gradient = np.zeros_like(points)
    
    for idx, (i, R) in enumerate(zip(valid_indices, radii)):
        simplex = hull.simplices[i]
        C = centers[idx]
        w_i = weights[idx]
        
        for v_idx in simplex:
            p = points[v_idx]
            dot_pc = np.dot(p, C)
            t = C - dot_pc * p
            norm_t = np.linalg.norm(t)
            if norm_t > 1e-10:
                t = t / norm_t
                gradient[v_idx] += w_i * R * t
    
    # Normalize per-point gradient to unit vectors for stable stepping
    grad_norm = np.linalg.norm(gradient, axis=1, keepdims=True)
    gradient = gradient / np.where(grad_norm < 1e-10, 1.0, grad_norm)
    
    return log_sum_exp, gradient

def _discrete_local_search(points, max_time, initial_alpha=0.05):
    """
    Discrete facet-based local search for high-precision polishing.
    Returns (best_points, best_radius_degrees).
    """
    best_points = points.copy()
    best_radius, _, _, _ = _compute_covering_radius(points)
    alpha = initial_alpha
    stagnation = 0
    max_stagnation = 50

    while time.time() < max_time:
        radius, facet_idx, center, hull = _compute_covering_radius(points)
        
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
            stagnation = 0
        else:
            stagnation += 1

        if center is None or facet_idx == -1:
            break

        simplex = hull.simplices[facet_idx]
        v0, v1, v2 = simplex
        improved = False

        # Try moving all 3 vertices together
        new_points = points.copy()
        new_points[v0] = _move_point_towards(points[v0], center, alpha)
        new_points[v1] = _move_point_towards(points[v1], center, alpha)
        new_points[v2] = _move_point_towards(points[v2], center, alpha)
        new_radius, _, _, _ = _compute_covering_radius(new_points)

        if new_radius < radius:
            points = new_points
            improved = True
        else:
            # Try individual vertex moves
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
            if alpha < 0.003:
                # Perturb to escape local minimum
                noise = np.random.randn(3, 3) * 0.03
                new_points = points.copy()
                new_points[v0] = points[v0] + noise[0]
                new_points[v1] = points[v1] + noise[1]
                new_points[v2] = points[v2] + noise[2]
                for idx in [v0, v1, v2]:
                    new_points[idx] = new_points[idx] / np.linalg.norm(new_points[idx])
                pert_radius, _, _, _ = _compute_covering_radius(new_points)
                
                if pert_radius < best_radius * 1.02:
                    points = new_points
                    alpha = 0.03
                    stagnation = 0
                elif stagnation > max_stagnation:
                    break
                else:
                    alpha = 0.015

    return best_points, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 142.0  # 8s safety margin before 150s kill

    np.random.seed(42)

    best_points = None
    best_radius = 90.0

    # Phase 1: Global optimization via repulsion from multiple starts
    for seed in range(6):
        if time.time() > deadline - 50:
            break

        np.random.seed(seed * 1000 + 42)

        if seed < 2:
            points = _spectral_init(N).copy()
        else:
            points = _fibonacci_spiral(N).copy()
        
        noise = np.random.randn(N, 3) * 0.04
        points = points + noise
        points = points / np.linalg.norm(points, axis=1, keepdims=True)

        # Global repulsion
        alpha = 0.02
        for step in range(200):
            if time.time() > deadline - 50:
                break
            points = _repulsion_step(points, alpha, iters=1)
            
            if step % 20 == 0 or step == 199:
                radius, _, _, _ = _compute_covering_radius(points)
                if radius < best_radius:
                    best_radius = radius
                    best_points = points.copy()

    # Phase 2: Smooth gradient-based optimization
    if best_points is not None:
        points = best_points.copy()
    else:
        points = _fibonacci_spiral(N).copy()
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        best_radius, _, _, _ = _compute_covering_radius(points)
        best_points = points.copy()

    alpha_smooth = 0.04
    beta = 40.0
    smooth_stagnation = 0
    max_smooth_stagnation = 80

    while time.time() < deadline - 25:
        smooth_r, gradient = _smooth_covering_radius_and_gradient(points, beta)
        
        # Step in gradient direction
        new_points = points + alpha_smooth * gradient
        new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
        
        # Check actual (discrete) radius
        actual_radius, _, _, _ = _compute_covering_radius(new_points)
        
        if actual_radius < best_radius:
            best_radius = actual_radius
            best_points = new_points.copy()
            points = new_points
            smooth_stagnation = 0
        else:
            smooth_stagnation += 1
            # Try smaller step
            if smooth_stagnation % 5 == 0:
                alpha_smooth *= 0.9
                if alpha_smooth < 0.002:
                    # Perturb
                    noise = np.random.randn(N, 3) * 0.02
                    perturbed = points + noise
                    perturbed = perturbed / np.linalg.norm(perturbed, axis=1, keepdims=True)
                    pert_radius, _, _, _ = _compute_covering_radius(perturbed)
                    if pert_radius < best_radius * 1.01:
                        points = perturbed
                        alpha_smooth = 0.02
                        smooth_stagnation = 0
                    elif smooth_stagnation > max_smooth_stagnation:
                        break
                    else:
                        alpha_smooth = 0.01

        # Adapt beta: increase as we get closer to convergence
        if smooth_r < 0.3:  # ~17 degrees
            beta = min(beta * 1.05, 200.0)

    # Phase 3: High-precision discrete polish
    if time.time() < deadline - 5:
        points = best_points.copy()
        best_points, best_radius = _discrete_local_search(points, deadline - 5, initial_alpha=0.03)

    return best_points
