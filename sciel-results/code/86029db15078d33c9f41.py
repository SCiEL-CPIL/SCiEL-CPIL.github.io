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
        Compute all facet circumcenters and their angular radii (vectorized).
        Returns: (radii_array, centers_array, simplices_array)
        """
        try:
            hull = ConvexHull(pts)
        except Exception:
            return np.array([]), np.array([]), np.array([])

        simplices = hull.simplices
        n_facets = len(simplices)
        if n_facets == 0:
            return np.array([]), np.array([]), np.array([])

        v1 = pts[simplices[:, 0]]
        v2 = pts[simplices[:, 1]]
        v3 = pts[simplices[:, 2]]

        edge1 = v2 - v1
        edge2 = v3 - v1
        normals = np.cross(edge1, edge2)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)

        valid = norms[:, 0] > 1e-12
        if not np.any(valid):
            return np.array([]), np.array([]), np.array([])

        centers = normals[valid] / norms[valid]
        # Ensure centers point outward (same side as vertices)
        dots = np.sum(centers * v1[valid], axis=1)
        centers[dots < 0] *= -1

        # Compute radii
        diffs = centers - v1[valid]
        d = np.linalg.norm(diffs, axis=1)
        radii = 2.0 * np.arcsin(np.clip(d / 2.0, 0.0, 1.0))

        return radii, centers, simplices[valid]

    def compute_covering_radius(pts):
        """
        Compute the covering radius in radians and the worst-case circumcenter.
        """
        radii, centers, _ = compute_facet_circumcenters(pts)
        if len(radii) == 0:
            return np.pi, np.array([0.0, 0.0, 1.0])

        idx = np.argmax(radii)
        return radii[idx], centers[idx]

    def grid_soft_max_objective(pts, grid_pts, beta=50.0):
        """
        Compute soft-max of angular distances from grid points to nearest configuration point.
        Returns: (soft_max_value, gradient_wrt_pts)
        """
        # Compute all pairwise angular distances: grid (M,3) x pts (N,3)
        # cos_theta = grid @ pts.T  -> (M, N)
        cos_theta = grid_pts @ pts.T  # (M, N)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angles = np.arccos(cos_theta)  # (M, N)

        # For each grid point, find the min angle to any config point
        min_angles = np.min(angles, axis=1)  # (M,)

        # Soft max of min_angles
        max_val = np.max(min_angles)
        shifted = beta * (min_angles - max_val)
        exp_terms = np.exp(shifted)
        log_sum = np.log(np.sum(exp_terms))
        soft_max_val = max_val + log_sum / beta
        weights = exp_terms / np.sum(exp_terms)  # (M,)

        # Gradient: d(soft_max)/d(pts)
        # For each grid point m, the nearest config point index
        nearest_idx = np.argmin(angles, axis=1)  # (M,)

        grad = np.zeros_like(pts)
        for m in range(len(grid_pts)):
            w = weights[m]
            if w < 1e-12:
                continue
            n_idx = nearest_idx[m]
            g = grid_pts[m]
            p = pts[n_idx]
            cos_t = np.dot(g, p)
            cos_t = np.clip(cos_t, -1.0, 1.0)
            sin_t = np.sqrt(np.clip(1.0 - cos_t * cos_t, 0.0, 1.0))
            if sin_t < 1e-10:
                continue
            # d(arccos(g.p))/dp = -g/sin_t
            grad_vec = -g / sin_t
            # Project to tangent space at p
            radial = np.dot(grad_vec, p) * p
            tangent_grad = grad_vec - radial
            grad[n_idx] += w * tangent_grad

        return soft_max_val, grad



    def spring_mass_relax(pts, iterations=500):
        """
        Spring-mass relaxation: treat convex hull edges as springs with rest length
        derived from optimal hexagonal packing. Forces vertices to minimize edge
        length variance while maintaining spherical constraint.
        """
        pts = pts.copy()
        s_opt = np.sqrt(8.0 * np.pi / (3.0 * np.sqrt(3.0) * N))
        
        damping = 0.1
        prev_energy = np.inf
        
        for it in range(iterations):
            if time.time() > deadline - 20:
                break
            
            try:
                hull = ConvexHull(pts)
            except:
                break
            
            edge_set = set()
            for simplex in hull.simplices:
                edge_set.add(tuple(sorted((simplex[0], simplex[1]))))
                edge_set.add(tuple(sorted((simplex[1], simplex[2]))))
                edge_set.add(tuple(sorted((simplex[2], simplex[0]))))
            
            edges = list(edge_set)
            if not edges:
                break
            
            forces = np.zeros_like(pts)
            for i, j in edges:
                v_i = pts[i]
                v_j = pts[j]
                cos_theta = np.clip(np.dot(v_i, v_j), -1.0, 1.0)
                theta = np.arccos(cos_theta)
                
                k = 1.0
                force_mag = k * (theta - s_opt)
                
                if theta < 1e-10:
                    continue
                dir_vec = v_j - np.dot(v_j, v_i) * v_i
                dir_norm = np.linalg.norm(dir_vec)
                if dir_norm < 1e-10:
                    continue
                dir_vec /= dir_norm
                
                forces[i] += force_mag * dir_vec
                forces[j] -= force_mag * dir_vec
            
            for i in range(N):
                v = pts[i]
                f_tangent = forces[i] - np.dot(forces[i], v) * v
                pts[i] = v + damping * f_tangent
                pts[i] /= np.linalg.norm(pts[i])
            
            current_energy = np.sum((np.linalg.norm(forces, axis=1))**2)
            if abs(prev_energy - current_energy) < 1e-12:
                break
            prev_energy = current_energy
        
        return pts

    def soft_max_radius_and_grad(pts, beta=50.0):
        """
        Compute soft-max radius and its gradient w.r.t. points using analytical gradients.
        Returns: (soft_radius, gradient_array)
        """
        radii, centers, simplices = compute_facet_circumcenters(pts)
        if len(radii) == 0:
            return np.pi, np.zeros_like(pts)

        # Log-sum-exp for numerical stability
        max_r = np.max(radii)
        shifted = beta * (radii - max_r)
        max_shifted = np.max(shifted)
        exp_terms = np.exp(shifted - max_shifted)
        log_sum = max_shifted + np.log(np.sum(exp_terms))
        soft_radius = max_r + log_sum / beta
        weights = exp_terms / np.sum(exp_terms)

        grads = np.zeros_like(pts)

        for i in range(len(radii)):
            w = weights[i]
            if w < 1e-15:
                continue
            r = radii[i]
            if r < 1e-12:
                continue

            C = centers[i]
            sin_r = np.sin(r)
            if sin_r < 1e-10:
                continue

            # Gradient of arccos(dot(C, p)) w.r.t. p is -C/sin(r)
            # Projected to tangent space at p: grad_t = grad - (grad . p) p
            simplex = simplices[i]
            p_pts = pts[simplex]

            grad_contrib = -C / sin_r

            dots = np.sum(p_pts * grad_contrib, axis=1)
            radial_comps = dots[:, np.newaxis] * p_pts
            tangential_grads = grad_contrib - radial_comps

            for k in range(3):
                grads[simplex[k]] += w * tangential_grads[k]

        return soft_radius, grads

    def sum_sq_min_dist_objective(params):
        """
        Objective for L-BFGS-B: smooth surrogate based on soft-max of facet circumradii.
        This directly targets minimizing the covering radius.
        """
        if time.time() > deadline - 10:
            return 1e10

        pts = params.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        pts = pts / norms

        radii, _, _ = compute_facet_circumcenters(pts)
        if len(radii) == 0:
            return 1e10

        tau = 0.015
        scaled = radii / tau
        max_scaled = np.max(scaled)
        log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        return tau * log_sum_exp

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

    # Build a dense spherical grid for the grid-based soft-max surrogate
    def build_dense_grid(n_rings=60):
        """Build a dense spherical grid for computing min-distance to nearest point."""
        grid_pts = []
        for ring in range(n_rings):
            theta = np.pi * (ring + 0.5) / n_rings  # polar angle
            sin_t = np.sin(theta)
            cos_t = np.cos(theta)
            n_theta = max(1, int(n_rings * sin_t * 1.5))
            for k in range(n_theta):
                phi = 2.0 * np.pi * k / n_theta
                x = sin_t * np.cos(phi)
                y = sin_t * np.sin(phi)
                z = cos_t
                grid_pts.append([x, y, z])
        grid_pts = np.array(grid_pts)
        norms = np.linalg.norm(grid_pts, axis=1, keepdims=True)
        grid_pts = grid_pts / norms
        return grid_pts

    # Phase 1: Initial Fibonacci spiral
    current = fibonacci_sphere(N)
    best = current.copy()
    best_radius, _ = compute_covering_radius(best)

    # Phase 2: Spring-mass relaxation to get a good global structure
    if time.time() < deadline - 40:
        try:
            spring_pts = spring_mass_relax(current, iterations=500)
            spring_radius, _ = compute_covering_radius(spring_pts)
            if spring_radius < best_radius:
                best_radius = spring_radius
                best = spring_pts.copy()
                current = spring_pts.copy()
        except Exception:
            pass

    # Phase 3: L-BFGS-B with analytical gradients on soft-max facet radii
    if time.time() < deadline - 25:
        try:
            x0 = current.flatten()
            # Use a schedule of increasing beta values to sharpen the approximation
            beta_schedule = [20.0, 50.0, 100.0, 200.0]
            for beta in beta_schedule:
                if time.time() > deadline - 25:
                    break

                def objective_and_grad(x, b=beta):
                    if time.time() > deadline - 10:
                        return 1e10, np.zeros_like(x)
                    pts = x.reshape(N, 3)
                    norms = np.linalg.norm(pts, axis=1, keepdims=True)
                    norms = np.where(norms < 1e-10, 1.0, norms)
                    pts = pts / norms
                    soft_r, grads = soft_max_radius_and_grad(pts, beta=b)
                    return soft_r, grads.reshape(-1)

                elapsed = time.time() - start_time
                remaining = deadline - 25 - elapsed
                if remaining < 5:
                    break
                max_iter = int(min(300, max(50, 100 * (remaining / 30.0))))

                result = minimize(
                    objective_and_grad,
                    x0,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': max_iter, 'maxfun': max_iter*2, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100}
                )

                if result.success or result.fun < objective_and_grad(x0)[0]:
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

    # Phase 5: Random restarts with small perturbations
    if time.time() < deadline - 25:
        np.random.seed(42)
        for restart in range(50):
            if time.time() > deadline - 25:
                break
            perturbed = best.copy()
            noise_scale = 0.02 / (restart + 1)
            for i in range(N):
                rand_vec = np.random.randn(3)
                rand_vec /= np.linalg.norm(rand_vec)
                perturbed[i] = perturbed[i] + noise_scale * rand_vec
                perturbed[i] /= np.linalg.norm(perturbed[i])
            for step in [0.01, 0.005, 0.002, 0.001]:
                if time.time() > deadline - 25:
                    break
                perturbed = optimize_worst_facet(perturbed, step, iterations=60)
            radius, _ = compute_covering_radius(perturbed)
            if radius < best_radius:
                best_radius = radius
                best = perturbed.copy()

    # Phase 6: Grid-based soft-max polish (differentiable surrogate via dense grid)
    if time.time() < deadline - 8:
        try:
            grid_pts = build_dense_grid(n_rings=80)
            x0 = best.flatten()

            def grid_obj_grad(x, b=80.0):
                if time.time() > deadline - 5:
                    return 1e10, np.zeros_like(x)
                pts = x.reshape(N, 3)
                norms = np.linalg.norm(pts, axis=1, keepdims=True)
                norms = np.where(norms < 1e-10, 1.0, norms)
                pts = pts / norms
                val, grad = grid_soft_max_objective(pts, grid_pts, beta=b)
                return val, grad.reshape(-1)

            # Try multiple beta values for the grid-based objective
            for b in [40.0, 80.0, 150.0]:
                if time.time() > deadline - 8:
                    break
                remaining = deadline - 8 - (time.time() - start_time)
                if remaining < 3:
                    break
                max_iter = int(min(200, max(30, 80 * (remaining / 10.0))))
                result = minimize(
                    grid_obj_grad,
                    x0,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': max_iter, 'maxfun': max_iter*2, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100}
                )
                if result.success or result.fun < grid_obj_grad(x0, b)[0]:
                    grid_pts_res = project_to_sphere(result.x.reshape(N, 3))
                    grid_radius, _ = compute_covering_radius(grid_pts_res)
                    if grid_radius < best_radius:
                        best_radius = grid_radius
                        best = grid_pts_res.copy()
                        x0 = best.flatten()
        except Exception:
            pass

    # Phase 7: Final high-precision polish using analytical gradients with high beta
    if time.time() < deadline - 3:
        final = best.copy()
        # Use very high beta to closely approximate the true max
        beta_final = 500.0

        def final_objective_and_grad(x, b=beta_final):
            if time.time() > deadline - 1:
                return 1e10, np.zeros_like(x)
            pts = x.reshape(N, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms = np.where(norms < 1e-10, 1.0, norms)
            pts = pts / norms
            soft_r, grads = soft_max_radius_and_grad(pts, beta=b)
            return soft_r, grads.reshape(-1)

        try:
            x0 = final.flatten()
            remaining = deadline - 3 - (time.time() - start_time)
            if remaining > 5:
                max_iter = int(min(500, max(100, 200 * (remaining / 10.0))))
                result = minimize(
                    final_objective_and_grad,
                    x0,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': max_iter, 'maxfun': max_iter*2, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 200}
                )
                if result.success:
                    polish_pts = project_to_sphere(result.x.reshape(N, 3))
                    polish_radius, _ = compute_covering_radius(polish_pts)
                    if polish_radius < best_radius:
                        best_radius = polish_radius
                        best = polish_pts.copy()
        except Exception:
            pass

        # Additional manual polish with tiny steps
        if time.time() < deadline - 1:
            final = best.copy()
            step = 0.0005
            prev_radius = compute_covering_radius(final)[0]
            stall_count = 0
            for it in range(3000):
                if time.time() > deadline - 1:
                    break
                radius, worst_center = compute_covering_radius(final)
                if radius >= prev_radius - 1e-16:
                    step *= 0.5
                    stall_count += 1
                    if step < 1e-16 or stall_count > 100:
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
