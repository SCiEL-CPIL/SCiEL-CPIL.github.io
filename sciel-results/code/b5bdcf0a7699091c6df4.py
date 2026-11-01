"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit (5s buffer)

    def fibonacci_sphere(n, offset=0.5, phase_shift=0.0):
        i = np.arange(n) + offset
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * (i + phase_shift)
        r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        return pts

    def compute_hull_data(pts):
        """
        Compute all facet circumcenters and their angular radii.
        Returns: (facets, circumcenters, radii) or (None, None, None) on failure.
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
            dots = np.sum(cc * a[valid], axis=1)
            cc[dots < 0] *= -1
            circumcenters[valid] = cc

            cos_r = np.clip(np.sum(cc * a[valid], axis=1), -1.0, 1.0)
            radii[valid] = np.arccos(cos_r)

        return facets, circumcenters, radii

    def compute_covering_radius(pts):
        """
        Compute the covering radius in radians and the worst-case circumcenter.
        """
        facets, circumcenters, radii = compute_hull_data(pts)
        if radii is None:
            return np.pi, np.array([0.0, 0.0, 1.0])

        max_idx = np.argmax(radii)
        max_radius = radii[max_idx]
        worst_center = circumcenters[max_idx]

        return max_radius, worst_center

    def soft_max_objective(params, beta=150.0):
        """
        Soft-max surrogate: approximates max(radii) via log-sum-exp.
        """
        if time.time() > deadline - 10:
            return 1e10

        pts = params.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        pts = pts / norms

        facets, circumcenters, radii = compute_hull_data(pts)
        if radii is None:
            return 1e10

        r_max = np.max(radii)
        if r_max < 1e-15:
            return r_max
        shifted = beta * (radii - r_max)
        log_sum_exp = np.log(np.sum(np.exp(shifted))) / beta + r_max
        return log_sum_exp

    def optimize_worst_facet(pts, step_size, iterations=60):
        """
        Iteratively move the vertices of the worst facet towards the worst center.
        Uses adaptive step size and tangent space projection.
        """
        pts = pts.copy()
        prev_radius = np.pi
        for _ in range(iterations):
            if time.time() > deadline - 5:
                break

            radius, worst_center = compute_covering_radius(pts)

            if radius >= prev_radius:
                step_size *= 0.5
                if step_size < 1e-8:
                    break
            else:
                prev_radius = radius

            facets, circumcenters, radii = compute_hull_data(pts)
            if facets is None:
                break

            # Find facets that are close to the worst center
            target_indices_set = set()
            for i in range(len(facets)):
                if np.dot(circumcenters[i], worst_center) > 0.999:
                    for idx in facets[i]:
                        target_indices_set.add(idx)

            if not target_indices_set:
                break

            for idx in target_indices_set:
                v = pts[idx]
                dir_vec = worst_center - v
                radial_comp = np.dot(dir_vec, v)
                dir_vec = dir_vec - radial_comp * v
                norm_dir = np.linalg.norm(dir_vec)
                if norm_dir > 1e-10:
                    dir_vec /= norm_dir
                    pts[idx] = v + step_size * dir_vec
                    pts[idx] /= np.linalg.norm(pts[idx])

        return pts

    def project_to_sphere(pts):
        """Project points onto the unit sphere."""
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        return pts / norms

    # Phase 1: Initial Fibonacci spiral
    current = fibonacci_sphere(N)
    best = current.copy()
    best_radius, _ = compute_covering_radius(best)

    # Phase 1.5: Repulsion relaxation to improve initial spread
    if time.time() < deadline - 80:
        def repulsion_objective(params):
            if time.time() > deadline - 80:
                return 1e10
            pts = params.reshape(N, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms = np.where(norms < 1e-10, 1.0, norms)
            pts = pts / norms
            
            d2 = np.sum(pts * pts[:, np.newaxis, :], axis=2)
            d2 = np.clip(d2, 1e-12, 4.0)
            d = np.sqrt(d2)
            np.fill_diagonal(d, 1.0)
            
            # Minimize -sum(log(d)) => Maximize sum(log(d))
            return -np.sum(np.log(d))

        try:
            x0 = current.flatten()
            result_rep = minimize(
                repulsion_objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 50, 'maxfun': 200, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            if result_rep.success or result_rep.fun < repulsion_objective(x0):
                rep_pts = project_to_sphere(result_rep.x.reshape(N, 3))
                rep_radius, _ = compute_covering_radius(rep_pts)
                if rep_radius < best_radius:
                    best_radius = rep_radius
                    best = rep_pts.copy()
                    current = rep_pts.copy()
        except Exception:
            pass

    # Phase 2: L-BFGS-B optimization on the soft-max objective
    if time.time() < deadline - 45:
        try:
            x0 = current.flatten()
            result = minimize(
                soft_max_objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 100, 'maxfun': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 100}
            )
            if result.fun < soft_max_objective(x0):
                lbfgs_pts = project_to_sphere(result.x.reshape(N, 3))
                lbfgs_radius, _ = compute_covering_radius(lbfgs_pts)
                if lbfgs_radius < best_radius:
                    best_radius = lbfgs_radius
                    best = lbfgs_pts.copy()
                    current = lbfgs_pts.copy()
        except Exception:
            pass

    # Phase 3: Worst-facet optimization with adaptive step size
    step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
    for step in step_sizes:
        if time.time() > deadline - 25:
            break

        current = optimize_worst_facet(current, step, iterations=80)
        radius, _ = compute_covering_radius(current)

        if radius < best_radius:
            best_radius = radius
            best = current.copy()
        else:
            current = best.copy()

    # Phase 4: Random restarts with small perturbations
    if time.time() < deadline - 30:
        np.random.seed(42)
        for restart in range(15):
            if time.time() > deadline - 30:
                break

            perturbed = best.copy()
            noise_scale = 0.015 / (restart + 1)
            for i in range(N):
                rand_vec = np.random.randn(3)
                rand_vec /= np.linalg.norm(rand_vec)
                perturbed[i] = perturbed[i] + noise_scale * rand_vec
                perturbed[i] /= np.linalg.norm(perturbed[i])

            for step in [0.01, 0.005, 0.002]:
                if time.time() > deadline - 30:
                    break
                perturbed = optimize_worst_facet(perturbed, step, iterations=40)

            radius, _ = compute_covering_radius(perturbed)
            if radius < best_radius:
                best_radius = radius
                best = perturbed.copy()

    # Phase 5: Final high-precision polish via adaptive step size in tangent space
    if time.time() < deadline - 3:
        final = best.copy()
        step = 0.002
        prev_radius = compute_covering_radius(final)[0]
        stall_count = 0
        for it in range(400):
            if time.time() > deadline - 5:
                break

            radius, worst_center = compute_covering_radius(final)

            if radius >= prev_radius - 1e-12:
                step *= 0.5
                stall_count += 1
                if step < 1e-12 or stall_count > 15:
                    break
            else:
                prev_radius = radius
                stall_count = 0

            facets, circumcenters, radii = compute_hull_data(final)
            if facets is None:
                break

            target_indices_set = set()
            for i in range(len(facets)):
                if np.dot(circumcenters[i], worst_center) > 0.9995:
                    for idx in facets[i]:
                        target_indices_set.add(idx)

            if not target_indices_set:
                break

            for idx in target_indices_set:
                v = final[idx]
                dir_vec = worst_center - v
                radial_comp = np.dot(dir_vec, v)
                dir_vec = dir_vec - radial_comp * v
                dir_vec_norm = np.linalg.norm(dir_vec)
                if dir_vec_norm > 1e-12:
                    dir_vec /= dir_vec_norm
                    final[idx] = v + step * dir_vec
                    final[idx] /= np.linalg.norm(final[idx])

        radius, _ = compute_covering_radius(final)
        if radius < best_radius:
            best_radius = radius
            best = final

    # Final normalization and validation
    best = project_to_sphere(best)

    norms = np.linalg.norm(best, axis=1)
    valid = (norms > 1e-10) & np.all(np.isfinite(best), axis=1)
    if not np.all(valid):
        best = fibonacci_sphere(N)

    return best
# EVOLVE-BLOCK-END
