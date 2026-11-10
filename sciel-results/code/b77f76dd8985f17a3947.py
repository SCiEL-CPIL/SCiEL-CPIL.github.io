import numpy as np
import time
from scipy.spatial import ConvexHull
import math

N = 130

def _compute_covering_radius_and_centers(points):
    """
    Computes the covering radius (max circumradius of hull facets)
    and returns the radius, circumcenter of worst facet, facet index, and hull.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return 90.0, None, -1, None

    simplices = hull.simplices
    max_radius = 0.0
    worst_center = None
    worst_facet_idx = -1

    for i, simplex in enumerate(simplices):
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
        if xT_M_x <= 1e-15:
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

    return math.degrees(max_radius), worst_center, worst_facet_idx, hull

def _move_point_towards(p, c, step_size):
    """Move point p on the sphere towards point c by step_size radians."""
    dot_cp = np.dot(c, p)
    t = c - dot_cp * p
    norm_t = np.linalg.norm(t)
    if norm_t < 1e-10:
        return p
    t = t / norm_t
    new_p = p + step_size * t
    new_p = new_p / np.linalg.norm(new_p)
    return new_p

def _repulsion_step(points, alpha):
    """One repulsion step to spread points apart."""
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
    forces = forces / np.where(forces_norm == 0, 1.0, forces_norm)
    new_points = points + alpha * forces
    new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
    return new_points

def _compute_lse_radius(points, sigma=0.01):
    """
    Compute a smooth approximation of the covering radius using log-sum-exp.
    This uses a grid of points on the sphere to estimate the max distance,
    but smoothed to allow gradient-like optimization if needed.
    For now, we use it as a secondary check or for a different optimization phase.
    """
    # Create a grid of points (Fibonacci sphere) for evaluation
    n_grid = 2000
    i = np.arange(n_grid) + 0.5
    z = 1.0 - 2.0 * i / n_grid
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    grid_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Compute angular distances from each grid point to each configuration point
    # dist[i, j] = arccos(dot(grid[i], config[j]))
    dots = grid_points @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)
    
    # For each grid point, find min distance to any config point
    min_dists = np.min(dists, axis=1)
    
    # Log-sum-exp approximation of the max
    # max(x) ~ (1/sigma) * log(sum(exp(sigma * x)))
    # To avoid overflow, subtract max
    max_d = np.max(min_dists)
    lse = (1.0 / sigma) * np.log(np.sum(np.exp(sigma * (min_dists - max_d)))) + max_d
    
    return math.degrees(lse)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def _time_ok():
        return time.time() < deadline

    # Phase 1: Multi-start initialization with Fibonacci spiral
    best_points = None
    best_radius = 180.0

    for attempt in range(3):
        if not _time_ok():
            break

        # Fibonacci spiral with different offsets
        i = np.arange(N) + 0.5 + attempt * 0.1
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

        # Add jitter
        np.random.seed(42 + attempt)
        noise = np.random.randn(N, 3) * 0.02
        points = points + noise
        points = points / np.linalg.norm(points, axis=1, keepdims=True)

        # Repulsion to spread
        alpha = 0.05
        for step in range(100):
            if not _time_ok():
                break
            points = _repulsion_step(points, alpha)
            alpha *= 0.98

        radius, _, _, _ = _compute_covering_radius_and_centers(points)
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()

    points = best_points.copy()

    # Phase 2: Local search targeting worst facet with adaptive step and perturbations
    alpha = 0.1
    stagnation = 0
    max_stagnation = 50

    for step in range(2000):
        if not _time_ok():
            break

        radius, center, facet_idx, hull = _compute_covering_radius_and_centers(points)

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

        # Try moving all 3 vertices
        new_points = points.copy()
        new_points[v0] = _move_point_towards(points[v0], center, alpha)
        new_points[v1] = _move_point_towards(points[v1], center, alpha)
        new_points[v2] = _move_point_towards(points[v2], center, alpha)
        new_radius, _, _, _ = _compute_covering_radius_and_centers(new_points)

        if new_radius < radius:
            points = new_points
        else:
            # Try individual moves
            improved = False
            for v_idx in [v0, v1, v2]:
                test_points = points.copy()
                test_points[v_idx] = _move_point_towards(points[v_idx], center, alpha)
                test_radius, _, _, _ = _compute_covering_radius_and_centers(test_points)
                if test_radius < radius:
                    points = test_points
                    improved = True
                    break

            if not improved:
                alpha *= 0.9
                if alpha < 0.001:
                    # Perturb to escape local minimum
                    # Use a larger perturbation and accept if it's not too bad
                    noise = np.random.randn(N, 3) * 0.05
                    perturbed = points + noise
                    perturbed = perturbed / np.linalg.norm(perturbed, axis=1, keepdims=True)
                    pert_radius, _, _, _ = _compute_covering_radius_and_centers(perturbed)
                    
                    # Accept if it improves best, or if it's close to current best
                    if pert_radius < best_radius * 1.02:
                        points = perturbed
                        if pert_radius < best_radius:
                            best_points = points.copy()
                            best_radius = pert_radius
                        alpha = 0.05
                    else:
                        # If stuck for too long, try a different strategy:
                        # Use the LSE radius to guide a global move?
                        # For now, just break if we've been stuck for a while
                        if stagnation > max_stagnation:
                            break
                        else:
                            # Try a smaller perturbation
                            alpha = 0.01

    # Final polish: Use a finer grid to check for any improvements
    # and potentially move points slightly to reduce the max radius
    if _time_ok():
        # Do a few more steps with very small alpha
        alpha = 0.005
        for step in range(100):
            if not _time_ok():
                break
            radius, center, facet_idx, hull = _compute_covering_radius_and_centers(points)
            if center is None or facet_idx == -1:
                break
            
            simplex = hull.simplices[facet_idx]
            v0, v1, v2 = simplex
            
            new_points = points.copy()
            new_points[v0] = _move_point_towards(points[v0], center, alpha)
            new_points[v1] = _move_point_towards(points[v1], center, alpha)
            new_points[v2] = _move_point_towards(points[v2], center, alpha)
            new_radius, _, _, _ = _compute_covering_radius_and_centers(new_points)
            
            if new_radius < best_radius:
                points = new_points
                best_points = points.copy()
                best_radius = new_radius
            else:
                break

    return best_points
