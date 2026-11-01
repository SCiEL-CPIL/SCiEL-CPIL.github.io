"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit

    def fibonacci_sphere(n, offset=0.5):
        i = np.arange(n) + offset
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        return pts

    def compute_facet_circumcenters(pts):
        """
        Compute all facet circumcenters and their angular radii (vectorized).
        Returns: (radii_array, centers_array, simplices_array)
        """
        try:
            hull = ConvexHull(pts)
        except Exception:
            return None, None, None

        simplices = hull.simplices  # shape (F, 3)
        F = len(simplices)
        
        # Vectorized: get all three vertices for each facet
        v1 = pts[simplices[:, 0]]  # (F, 3)
        v2 = pts[simplices[:, 1]]  # (F, 3)
        v3 = pts[simplices[:, 2]]  # (F, 3)

        # Compute normals
        edge1 = v2 - v1  # (F, 3)
        edge2 = v3 - v1  # (F, 3)
        normals = np.cross(edge1, edge2)  # (F, 3)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)  # (F, 1)
        
        # Filter out degenerate facets
        valid = norms.squeeze() > 1e-12
        if not np.any(valid):
            return None, None, None
        
        normals = normals[valid]
        v1 = v1[valid]
        simplices_valid = simplices[valid]
        
        # Normalize normals
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        centers = normals / norms  # (F, 3)
        
        # Ensure centers point outward (dot with v1 > 0)
        dots = np.sum(centers * v1, axis=1)  # (F,)
        centers[dots < 0] *= -1
        
        # Compute radii: radius = 2 * arcsin(|c - v1| / 2)
        diffs = centers - v1  # (F, 3)
        d = np.linalg.norm(diffs, axis=1)  # (F,)
        radii = 2.0 * np.arcsin(np.clip(d / 2.0, 0.0, 1.0))  # (F,)
        
        return radii, centers, simplices_valid

    def compute_covering_radius(pts):
        """
        Compute the covering radius in radians and the worst-case circumcenter.
        """
        radii, centers, simplices = compute_facet_circumcenters(pts)
        if radii is None:
            return np.pi, np.array([0.0, 0.0, 1.0])

        max_idx = np.argmax(radii)
        max_radius = radii[max_idx]
        worst_center = centers[max_idx]

        return max_radius, worst_center

    def sum_sq_min_dist_objective(params):
        """
        Objective for L-BFGS-B: smooth surrogate based on maximum facet radius.
        Uses a soft-max over squared facet radii to encourage uniformity.
        """
        if time.time() > deadline - 10:
            return 1e10

        pts = params.reshape(N, 3)
        # Normalize
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        pts = pts / norms

        radii, centers, simplices = compute_facet_circumcenters(pts)
        if radii is None:
            return 1e10

        # Soft-max over squared radii: tau * log(sum(exp(r^2/tau)))
        # This is a smooth approximation to max(r^2)
        tau = 0.02
        r2 = radii ** 2
        r2_max = np.max(r2)
        log_sum = r2_max + tau * np.log(np.sum(np.exp((r2 - r2_max) / tau)))
        soft_max_r2 = log_sum
        
        # Also add a small penalty for variance to encourage uniformity
        var_penalty = 0.01 * np.var(r2)
        
        return soft_max_r2 + var_penalty

    def max_angular_distance_objective(params):
        """
        Objective for final polish: the maximum angular distance (covering radius).
        """
        if time.time() > deadline - 5:
            return 1e10

        pts = params.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms

        radius, _ = compute_covering_radius(pts)
        return radius

    def optimize_worst_facet(pts, step_size, iterations=50):
        """
        Iteratively move the vertices of the worst facet towards the worst center.
        Also considers nearby facets to avoid local minima.
        Uses adaptive step size and tangent space projection.
        """
        pts = pts.copy()
        prev_radius = np.pi
        for _ in range(iterations):
            if time.time() > deadline - 5:
                break

            radius, worst_center = compute_covering_radius(pts)

            # If radius hasn't improved, reduce step size
            if radius >= prev_radius - 1e-14:
                step_size *= 0.7
                if step_size < 1e-10:
                    break
            else:
                prev_radius = radius

            # Get facet data (vectorized)
            radii, centers, simplices = compute_facet_circumcenters(pts)
            if radii is None:
                break

            # Find facets that are close to the worst center (within threshold)
            # Use dot product to find similar directions
            dots = np.sum(centers * worst_center, axis=1)  # (F,)
            mask = dots > 0.998  # Slightly looser threshold to include nearby facets
            target_indices = set()
            for simplex in simplices[mask]:
                for idx in simplex:
                    target_indices.add(int(idx))

            if not target_indices:
                break

            # Move vertices towards worst_center in tangent space
            for idx in target_indices:
                v = pts[idx]
                dir_vec = worst_center - v
                # Project onto tangent space: remove radial component
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

    # Phase 2: L-BFGS-B optimization on the smooth surrogate objective
    if time.time() < deadline - 30:
        try:
            x0 = current.flatten()
            result = minimize(
                sum_sq_min_dist_objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 800, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if result.success or result.fun < sum_sq_min_dist_objective(x0):
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
        if time.time() > deadline - 15:
            break

        current = optimize_worst_facet(current, step, iterations=100)
        radius, _ = compute_covering_radius(current)

        if radius < best_radius:
            best_radius = radius
            best = current.copy()
        else:
            current = best.copy()

    # Phase 4: Random restarts with small perturbations
    if time.time() < deadline - 20:
        np.random.seed(42)  # For reproducibility
        for restart in range(50):
            if time.time() > deadline - 20:
                break

            # Create a perturbed version of the best solution
            # Use a wider range of noise scales
            noise_scale = 0.02 * np.exp(-restart * 0.05)
            perturbed = best.copy()
            for i in range(N):
                rand_vec = np.random.randn(3)
                rand_vec /= np.linalg.norm(rand_vec)
                perturbed[i] = perturbed[i] + noise_scale * rand_vec
                perturbed[i] /= np.linalg.norm(perturbed[i])

            # Optimize the perturbed solution
            for step in [0.015, 0.008, 0.004, 0.002]:
                if time.time() > deadline - 20:
                    break
                perturbed = optimize_worst_facet(perturbed, step, iterations=50)

            radius, _ = compute_covering_radius(perturbed)
            if radius < best_radius:
                best_radius = radius
                best = perturbed.copy()

    # Phase 5: Final high-precision polish via adaptive step size in tangent space
    if time.time() < deadline - 5:
        final = best.copy()
        # Use adaptive step size that decreases as we converge
        step = 0.003
        prev_radius = compute_covering_radius(final)[0]
        stall_count = 0
        for it in range(800):
            if time.time() > deadline - 5:
                break

            radius, worst_center = compute_covering_radius(final)

            # If no improvement, reduce step size
            if radius >= prev_radius - 1e-14:
                step *= 0.6
                stall_count += 1
                if step < 1e-13 or stall_count > 30:
                    break
            else:
                prev_radius = radius
                stall_count = 0

            # Get facet data (vectorized)
            radii, centers, simplices = compute_facet_circumcenters(final)
            if radii is None:
                break

            # Find all facets close to worst_center (use tighter threshold)
            dots = np.sum(centers * worst_center, axis=1)
            mask = dots > 0.9995
            target_indices_set = set()
            for simplex in simplices[mask]:
                for idx in simplex:
                    target_indices_set.add(int(idx))

            if not target_indices_set:
                break

            for idx in target_indices_set:
                v = final[idx]
                dir_vec = worst_center - v
                # Project onto tangent space: remove radial component
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

    # Ensure all points are non-zero and finite
    norms = np.linalg.norm(best, axis=1)
    valid = (norms > 1e-10) & np.all(np.isfinite(best), axis=1)
    if not np.all(valid):
        # Fallback to Fibonacci if something went wrong
        best = fibonacci_sphere(N)

    return best
# EVOLVE-BLOCK-END
