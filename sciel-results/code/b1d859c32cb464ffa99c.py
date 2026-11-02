"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def fibonacci_sphere(n, offset=0.5):
        i = np.arange(n) + offset
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        return pts

    def compute_facet_circumcenters(pts):
        """
        Compute all facet circumcenters and their angular radii.
        Returns: list of (radius_radians, circumcenter, simplex_indices)
        """
        try:
            hull = ConvexHull(pts)
        except Exception:
            return []

        results = []
        for simplex in hull.simplices:
            v1 = pts[simplex[0]]
            v2 = pts[simplex[1]]
            v3 = pts[simplex[2]]

            edge1 = v2 - v1
            edge2 = v3 - v1
            n = np.cross(edge1, edge2)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-12:
                continue

            c = n / norm_n
            if np.dot(c, v1) < 0:
                c = -c

            d = np.linalg.norm(c - v1)
            radius = 2.0 * np.arcsin(np.clip(d / 2.0, 0.0, 1.0))
            results.append((radius, c, simplex))

        return results

    def compute_covering_radius(pts):
        """
        Compute the covering radius in radians and the worst-case circumcenter.
        """
        facets = compute_facet_circumcenters(pts)
        if not facets:
            return np.pi, np.array([0.0, 0.0, 1.0])

        max_radius = 0.0
        worst_center = None
        for radius, center, _ in facets:
            if radius > max_radius:
                max_radius = radius
                worst_center = center

        return max_radius, worst_center

    def soft_max_grid_objective_and_grad(params, tau=0.02):
        """
        Analytical soft-max grid objective and gradient.
        Returns (value, gradient) for L-BFGS-B with jac.
        """
        if time.time() > deadline - 10:
            return 1e10, np.zeros_like(params)

        pts = params.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        pts = pts / norms

        # Use a fine grid of evaluation points
        n_grid = 400
        grid_pts = fibonacci_sphere(n_grid)

        # Compute angular distances from each grid point to all configuration points
        dots = grid_pts @ pts.T  # shape (n_grid, N)
        cos_theta = np.clip(dots, -1.0, 1.0)
        theta = np.arccos(cos_theta)  # angular distances (n_grid, N)

        # For each grid point, find the minimum distance to any configuration point
        min_dist = np.min(theta, axis=1)  # shape (n_grid,)
        min_idx = np.argmin(theta, axis=1)  # which config point is nearest

        # Soft-max over the grid points
        scaled = min_dist / tau
        max_scaled = np.max(scaled)
        if max_scaled > 500:
            val = float(max_scaled * tau)
            # Approximate gradient when overflow
            grad = np.zeros_like(pts)
            for g in range(n_grid):
                idx = min_idx[g]
                grad[idx] += grid_pts[g] * (1.0 / n_grid)
            return val, grad.flatten()

        exp_scaled = np.exp(scaled - max_scaled)
        sum_exp = np.sum(exp_scaled)
        log_sum_exp = max_scaled + np.log(sum_exp)
        val = tau * log_sum_exp

        # Gradient: d/dparams [tau * log(sum(exp(min_dist/tau)))]
        # = sum_g [exp(scaled_g) / sum_exp] * (1/tau) * d(min_dist_g)/dparams
        # d(min_dist_g)/dparams = d(theta_g,k_g)/dparams where k_g = min_idx[g]
        # theta_g,k = arccos(cos_theta_g,k)
        # d(arccos(x))/dx = -1/sqrt(1-x^2)
        # d(cos_theta_g,k)/d(pts[i]) = -grid_pts[g] / sqrt(1-cos_theta^2) for i=k_g, else 0
        # So d(theta_g,k_g)/d(pts[k_g]) = grid_pts[g] (tangent component)
        
        weights = exp_scaled / sum_exp  # shape (n_grid,)
        grad = np.zeros_like(pts)
        
        for g in range(n_grid):
            k = min_idx[g]
            w = weights[g]
            # d(theta)/d(pts[k]) = grid_pts[g] projected onto tangent space at pts[k]
            # = grid_pts[g] - (grid_pts[g]·pts[k]) * pts[k]
            proj = grid_pts[g] - np.dot(grid_pts[g], pts[k]) * pts[k]
            grad[k] += w * proj

        # Scale by 1/tau from chain rule: d(soft_max)/dparams = sum(w_g * d(min_dist_g)/dparams) / tau * tau... 
        # Actually: soft_max = tau * log(sum(exp(min_dist/tau)))
        # d(soft_max)/dparams = sum_g [exp(min_dist_g/tau) / sum_exp] * (1/tau) * d(min_dist_g)/dparams * tau
        # = sum_g [w_g] * d(min_dist_g)/dparams
        # So no extra 1/tau factor needed beyond what's in weights
        
        return val, grad.flatten()

    def facet_soft_max_objective(params, tau=0.015):
        """
        Epigraph-style soft-max of facet circumradii with analytical gradient.
        Uses auxiliary variable t for smoother landscape.
        """
        if time.time() > deadline - 10:
            return 1e10, np.zeros_like(params)

        pts = params.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        pts = pts / norms

        try:
            hull = ConvexHull(pts)
        except Exception:
            return 1e10, np.zeros_like(params)

        radii = []
        centers = []
        simplices = []
        for simplex in hull.simplices:
            v1 = pts[simplex[0]]
            v2 = pts[simplex[1]]
            v3 = pts[simplex[2]]
            edge1 = v2 - v1
            edge2 = v3 - v1
            n = np.cross(edge1, edge2)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-12:
                continue
            c = n / norm_n
            if np.dot(c, v1) < 0:
                c = -c
            d = np.linalg.norm(c - v1)
            radius = 2.0 * np.arcsin(np.clip(d / 2.0, 0.0, 1.0))
            radii.append(radius)
            centers.append(c)
            simplices.append(simplex)

        if not radii:
            return 1e10, np.zeros_like(params)

        radii_arr = np.array(radii)
        scaled = radii_arr / tau
        max_scaled = np.max(scaled)
        if max_scaled > 500:
            val = float(max_scaled * tau)
            # Find worst facet and move its vertices toward center
            worst_idx = np.argmax(radii_arr)
            grad = np.zeros_like(pts)
            c = centers[worst_idx]
            for idx in simplices[worst_idx]:
                v = pts[idx]
                dir_vec = c - v
                radial_comp = np.dot(dir_vec, v)
                dir_vec = dir_vec - radial_comp * v
                dir_norm = np.linalg.norm(dir_vec)
                if dir_norm > 1e-10:
                    grad[idx] += dir_vec / dir_norm
            return val, grad.flatten()

        exp_scaled = np.exp(scaled - max_scaled)
        sum_exp = np.sum(exp_scaled)
        log_sum_exp = max_scaled + np.log(sum_exp)
        val = tau * log_sum_exp

        weights = exp_scaled / sum_exp  # shape (n_facets,)
        grad = np.zeros_like(pts)
        
        for f_idx in range(len(radii)):
            w = weights[f_idx]
            if w < 1e-10:
                continue
            c = centers[f_idx]
            simplex = simplices[f_idx]
            # d(radius)/d(pts[i]) for i in simplex
            # radius = 2*arcsin(d/2) where d = |c - v1|
            # d(radius)/d(c) = (c - v1) / |c - v1| / cos(arcsin(d/2)) = (c-v1)/d / sqrt(1-d^2/4)
            # d(c)/d(pts[i]) is complex; approximate by moving toward/away from c
            for idx in simplex:
                v = pts[idx]
                # Move vertex toward circumcenter reduces radius
                dir_vec = c - v
                radial_comp = np.dot(dir_vec, v)
                dir_vec = dir_vec - radial_comp * v
                dir_norm = np.linalg.norm(dir_vec)
                if dir_norm > 1e-10:
                    grad[idx] += w * dir_vec / dir_norm

        return val, grad.flatten()

    def optimize_worst_facet(pts, step_size, iterations=50):
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

            try:
                hull = ConvexHull(pts)
            except:
                break

            facet_data = []
            for simplex in hull.simplices:
                v1 = pts[simplex[0]]
                v2 = pts[simplex[1]]
                v3 = pts[simplex[2]]
                edge1 = v2 - v1
                edge2 = v3 - v1
                n = np.cross(edge1, edge2)
                norm_n = np.linalg.norm(n)
                if norm_n < 1e-12:
                    continue
                c = n / norm_n
                if np.dot(c, v1) < 0:
                    c = -c
                facet_data.append((c, simplex))

            if not facet_data:
                break

            target_indices_set = set()
            for c, simplex in facet_data:
                if np.dot(c, worst_center) > 0.999:
                    for idx in simplex:
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

    # Phase 2: Soft-max grid objective optimization with analytical gradient
    if time.time() < deadline - 40:
        try:
            x0 = current.flatten()
            val0, _ = soft_max_grid_objective_and_grad(x0, 0.03)
            result_sm = minimize(
                lambda x: soft_max_grid_objective_and_grad(x, 0.03)[0],
                x0,
                jac=lambda x: soft_max_grid_objective_and_grad(x, 0.03)[1],
                method='L-BFGS-B',
                options={'maxiter': 400, 'maxfun': 3000, 'ftol': 1e-12, 'gtol': 1e-9}
            )
            val_sm, _ = soft_max_grid_objective_and_grad(result_sm.x, 0.03)
            if result_sm.success or val_sm < val0:
                sm_pts = project_to_sphere(result_sm.x.reshape(N, 3))
                sm_radius, _ = compute_covering_radius(sm_pts)
                if sm_radius < best_radius:
                    best_radius = sm_radius
                    best = sm_pts.copy()
                    current = sm_pts.copy()
        except Exception:
            pass

    # Phase 3: L-BFGS-B on facet soft-max with analytical gradient (tighter tau schedule)
    if time.time() < deadline - 25:
        try:
            x0 = current.flatten()
            for schedule_idx in range(4):
                if time.time() > deadline - 25:
                    break
                tau_val = 0.015 / (schedule_idx + 1)
                ftol = 1e-11 / (schedule_idx + 1)
                gtol = 1e-8 / (schedule_idx + 1)
                val0, _ = facet_soft_max_objective(x0, tau_val)
                result = minimize(
                    lambda x, tv=tau_val: facet_soft_max_objective(x, tv)[0],
                    x0,
                    jac=lambda x, tv=tau_val: facet_soft_max_objective(x, tv)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 400, 'maxfun': 3000, 'ftol': ftol, 'gtol': gtol}
                )
                val_res, _ = facet_soft_max_objective(result.x, tau_val)
                if result.success or val_res < val0:
                    lbfgs_pts = project_to_sphere(result.x.reshape(N, 3))
                    lbfgs_radius, _ = compute_covering_radius(lbfgs_pts)
                    if lbfgs_radius < best_radius:
                        best_radius = lbfgs_radius
                        best = lbfgs_pts.copy()
                        current = lbfgs_pts.copy()
                        x0 = best.flatten()
        except Exception:
            pass

    # Phase 4: Worst-facet optimization with adaptive step size
    step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
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

    # Phase 5: Random restarts with small perturbations + facet-targeted refinement
    if time.time() < deadline - 20:
        np.random.seed(42)
        for restart in range(60):
            if time.time() > deadline - 20:
                break
            perturbed = best.copy()
            noise_scale = 0.010 / (restart + 1)
            for i in range(N):
                rand_vec = np.random.randn(3)
                rand_vec /= np.linalg.norm(rand_vec)
                perturbed[i] = perturbed[i] + noise_scale * rand_vec
                perturbed[i] /= np.linalg.norm(perturbed[i])
            # First do worst-facet optimization
            for step in [0.01, 0.005, 0.002, 0.001]:
                if time.time() > deadline - 20:
                    break
                perturbed = optimize_worst_facet(perturbed, step, iterations=40)
            # Then refine with facet soft-max L-BFGS-B
            if time.time() < deadline - 22:
                try:
                    x0 = perturbed.flatten()
                    val0, _ = facet_soft_max_objective(x0, 0.01)
                    result_ref = minimize(
                        lambda x: facet_soft_max_objective(x, 0.01)[0],
                        x0,
                        jac=lambda x: facet_soft_max_objective(x, 0.01)[1],
                        method='L-BFGS-B',
                        options={'maxiter': 150, 'maxfun': 1200, 'ftol': 1e-11, 'gtol': 1e-8}
                    )
                    val_res, _ = facet_soft_max_objective(result_ref.x, 0.01)
                    if result_ref.success or val_res < val0:
                        ref_pts = project_to_sphere(result_ref.x.reshape(N, 3))
                        ref_radius, _ = compute_covering_radius(ref_pts)
                        if ref_radius < compute_covering_radius(perturbed)[0]:
                            perturbed = ref_pts
                except Exception:
                    pass
            radius, _ = compute_covering_radius(perturbed)
            if radius < best_radius:
                best_radius = radius
                best = perturbed.copy()

    # Phase 6: Final high-precision polish with facet-targeted refinement
    if time.time() < deadline - 5:
        final = best.copy()
        # First, do a final facet soft-max L-BFGS-B pass with very tight parameters
        if time.time() < deadline - 8:
            try:
                x0 = final.flatten()
                val0, _ = facet_soft_max_objective(x0, 0.005)
                result_final = minimize(
                    lambda x: facet_soft_max_objective(x, 0.005)[0],
                    x0,
                    jac=lambda x: facet_soft_max_objective(x, 0.005)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'maxfun': 2500, 'ftol': 1e-13, 'gtol': 1e-10}
                )
                val_res, _ = facet_soft_max_objective(result_final.x, 0.005)
                if result_final.success or val_res < val0:
                    fin_pts = project_to_sphere(result_final.x.reshape(N, 3))
                    fin_radius, _ = compute_covering_radius(fin_pts)
                    if fin_radius < compute_covering_radius(final)[0]:
                        final = fin_pts
            except Exception:
                pass

        # Then do worst-facet polish
        step = 0.0005
        prev_radius = compute_covering_radius(final)[0]
        stall_count = 0
        for it in range(500):
            if time.time() > deadline - 5:
                break
            radius, worst_center = compute_covering_radius(final)
            if radius >= prev_radius - 1e-16:
                step *= 0.7
                stall_count += 1
                if step < 1e-16 or stall_count > 40:
                    break
            else:
                prev_radius = radius
                stall_count = 0

            try:
                hull = ConvexHull(final)
            except:
                break

            target_indices_set = set()
            for simplex in hull.simplices:
                v1 = final[simplex[0]]
                v2 = final[simplex[1]]
                v3 = final[simplex[2]]
                edge1 = v2 - v1
                edge2 = v3 - v1
                n = np.cross(edge1, edge2)
                norm_n = np.linalg.norm(n)
                if norm_n < 1e-12:
                    continue
                c = n / norm_n
                if np.dot(c, v1) < 0:
                    c = -c
                if np.dot(c, worst_center) > 0.99999:
                    for idx in simplex:
                        target_indices_set.add(idx)

            if not target_indices_set:
                break

            for idx in target_indices_set:
                v = final[idx]
                dir_vec = worst_center - v
                radial_comp = np.dot(dir_vec, v)
                dir_vec = dir_vec - radial_comp * v
                dir_vec_norm = np.linalg.norm(dir_vec)
                if dir_vec_norm > 1e-13:
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
