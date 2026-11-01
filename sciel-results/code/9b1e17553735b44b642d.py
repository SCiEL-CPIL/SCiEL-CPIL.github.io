"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # 1. Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def get_hull_data(pts):
        """
        Computes convex hull facets, circumcenters, and radii (vectorized).
        Returns (facets, circumcenters, radii).
        """
        try:
            hull = ConvexHull(pts)
            facets = hull.simplices
        except Exception:
            return None, None, None

        n_facets = len(facets)
        if n_facets == 0:
            return None, None, None

        a = pts[facets[:, 0]]
        b = pts[facets[:, 1]]
        c = pts[facets[:, 2]]

        v1 = a - b
        v2 = b - c
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n, axis=1)

        valid = norm_n > 1e-12
        circumcenters = np.zeros((n_facets, 3))
        radii = np.zeros(n_facets)

        if np.any(valid):
            cc = n[valid] / norm_n[valid, np.newaxis]
            # Ensure circumcenter faces outward (toward 'a')
            dots = np.sum(cc * a[valid], axis=1)
            cc[dots < 0] *= -1
            circumcenters[valid] = cc

            cos_r = np.clip(np.sum(cc * a[valid], axis=1), -1.0, 1.0)
            radii[valid] = np.arccos(cos_r)

        return facets, circumcenters, radii

    def get_covering_info(pts):
        """
        Returns the covering radius (max facet radius) and the worst circumcenter.
        """
        facets, circumcenters, radii = get_hull_data(pts)
        if radii is None or len(radii) == 0:
            return 0.0, np.random.randn(3) / np.linalg.norm(np.random.randn(3))

        max_idx = np.argmax(radii)
        return radii[max_idx], circumcenters[max_idx]

    def calc_covering_radius(pts):
        radius, _ = get_covering_info(pts)
        return radius

    # Phase 1: Repulsion Relaxation (Thomson Problem style)
    # This spreads points evenly to get a good starting point for the covering optimization.
    current_points = points.copy()
    alpha = 0.05
    decay = 0.995
    for it in range(200):
        if time.time() > deadline - 20:
            break

        # Calculate forces: F_i = sum_j (p_i - p_j) / ||p_i - p_j||^2
        diff = current_points[:, np.newaxis, :] - current_points[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)

        mask = np.eye(N, dtype=bool)
        inv_dist_sq = np.zeros_like(dist_sq)
        np.divide(1.0, dist_sq, out=inv_dist_sq, where=~mask)

        forces = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)

        # Project onto tangent plane
        dot_fp = np.sum(forces * current_points, axis=1)
        forces_tan = forces - dot_fp[:, np.newaxis] * current_points

        # Update
        current_points = current_points + alpha * forces_tan
        norms = np.linalg.norm(current_points, axis=1)
        current_points = current_points / norms[:, np.newaxis]

        alpha *= decay

    best_points = current_points.copy()
    best_radius = calc_covering_radius(current_points)

    # Phase 2: L-BFGS optimization with soft-max surrogate
    # Objective: min max(radii) approximated by log-sum-exp(beta * (radii - max))
    # We use a fixed beta to make the objective smooth and differentiable.
    # The gradient is computed numerically because the hull structure changes discretely.
    # However, for a fixed set of active facets (the ones with large radius), the gradient
    # can be derived. Here we use a simple numerical gradient or a heuristic gradient.
    # A better approach: use the fact that the covering radius is determined by the "worst" facet.
    # If we assume the worst facet remains the worst, we can derive a gradient.
    # But the worst facet changes.
    # Alternative: Use a soft-max objective and compute its gradient numerically.
    # Given the time limit, we will use a simpler gradient descent step on the tangent space
    # based on the worst facet's circumcenter.

    # Let's try a hybrid:
    # 1. Identify the worst facet F* with circumcenter C*.
    # 2. Move the vertices of F* towards C* to reduce its radius.
    # 3. Move other points slightly away from C* to prevent them from becoming worse?
    #    Actually, moving vertices of F* towards C* is the standard "gradient" step for the max term.
    #    For the other points, we want to keep them in place or move them to maintain balance.
    #    A simple approach: Move ALL points slightly towards their "Voronoi" center (Lloyd's).
    #    But Lloyd's is expensive.
    #    Let's stick to the "worst facet" gradient but apply it with a decaying step size,
    #    and periodically re-evaluate.

    step_size = 0.1
    decay_step = 0.99
    current_points = best_points.copy()

    for it in range(500):
        if time.time() > deadline - 10:
            break

        radius, worst_center = get_covering_info(current_points)

        if radius < best_radius:
            best_radius = radius
            best_points = current_points.copy()

        # Find the facet corresponding to worst_center
        facets, circumcenters, radii = get_hull_data(current_points)
        if facets is None:
            break

        # Find the facet index
        target_idx = None
        for i, cc in enumerate(circumcenters):
            if np.dot(cc, worst_center) > 0.9999:
                target_idx = i
                break

        if target_idx is None:
            # If we can't find it, maybe the hull changed significantly.
            # Try to find the facet with max radius again
            target_idx = np.argmax(radii)

        target_facet = facets[target_idx]

        # Move the vertices of the target facet towards worst_center
        for idx in target_facet:
            v = current_points[idx]
            dir_vec = worst_center - v
            norm_dir = np.linalg.norm(dir_vec)
            if norm_dir > 1e-10:
                dir_vec /= norm_dir
            current_points[idx] = v + step_size * dir_vec
            current_points[idx] /= np.linalg.norm(current_points[idx])

        # Add a small repulsive force to all points to prevent clustering
        # and to ensure we don't get stuck in a local minimum where points are too close.
        # This is a simplified version of the repulsion phase.
        diff = current_points[:, np.newaxis, :] - current_points[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        mask = np.eye(N, dtype=bool)
        inv_dist_sq = np.zeros_like(dist_sq)
        np.divide(1.0, dist_sq, out=inv_dist_sq, where=~mask)
        forces = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        dot_fp = np.sum(forces * current_points, axis=1)
        forces_tan = forces - dot_fp[:, np.newaxis] * current_points

        # Blend the gradient step with the repulsion step
        # We already applied the gradient step above. Now apply a small repulsion step.
        repulse_step = 0.01
        current_points = current_points + repulse_step * forces_tan
        norms = np.linalg.norm(current_points, axis=1)
        current_points = current_points / norms[:, np.newaxis]

        step_size *= decay_step

        if step_size < 0.001:
            # Add small noise to escape local minima
            current_points += np.random.randn(N, 3) * 0.01
            current_points /= np.linalg.norm(current_points, axis=1, keepdims=True)
            step_size = 0.05

    # Phase 3: Final Polish with L-BFGS
    # Use a soft-max objective to make it smooth for L-BFGS.
    # Objective: log-sum-exp(beta * radii)
    # We will use a fixed beta and let L-BFGS minimize it.
    # The gradient is computed numerically.

    def soft_max_objective(params, beta=50.0):
        X = params.reshape(N, 3)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        X = X / norms

        facets, circumcenters, radii = get_hull_data(X)
        if radii is None or len(radii) == 0:
            return 1e10

        r_max = np.max(radii)
        if r_max < 1e-15:
            return r_max

        # Soft-max: log(sum(exp(beta * (radii - r_max)))) / beta + r_max
        # This is the same as log-sum-exp(beta * radii) / beta
        # But we shift by r_max for numerical stability.
        shifted = beta * (radii - r_max)
        log_sum_exp = np.log(np.sum(np.exp(shifted))) / beta + r_max
        return log_sum_exp

    # We will use L-BFGS-B on the flattened array of points.
    # The constraint is that each point is on the unit sphere.
    # L-BFGS-B doesn't handle equality constraints well directly.
    # Instead, we can parameterize the sphere using coordinates on the tangent space,
    # but that's complex.
    # Alternative: Use the fact that the objective is invariant to scaling, so we can
    # just minimize the objective and then normalize the result.
    # But L-BFGS might move points off the sphere.
    # A better approach: Use a penalty method or a custom optimizer.
    # Given the time limit, we will use a simple gradient descent with projection onto the sphere.
    # We already did something similar in Phase 2.
    # Let's try a more aggressive local search around the best point found so far.

    if time.time() < deadline - 5:
        current_best = best_points.copy()
        current_score = best_radius
        step = 0.05

        for i in range(200):
            if time.time() > deadline - 5:
                break
            improved = False
            for j in range(N):
                if time.time() > deadline - 5:
                    break
                for _ in range(5):
                    rand_vec = np.random.randn(3)
                    rand_vec /= np.linalg.norm(rand_vec)

                    old_p = current_best[j].copy()
                    new_p = current_best[j] + step * rand_vec
                    new_p /= np.linalg.norm(new_p)

                    current_best[j] = new_p
                    new_score = calc_covering_radius(current_best)

                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                    else:
                        current_best[j] = old_p

            if not improved:
                break

        if current_score < best_radius:
            best_points = current_best
            best_radius = current_score

    # Final normalization
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)

    return best_points
# EVOLVE-BLOCK-END
