"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 143.0  # Safety margin for 150s limit

    def fibonacci_sphere(n, offset=0.5):
        i = np.arange(n) + offset
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        return pts

    def project_to_sphere(pts):
        """Project points onto the unit sphere."""
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        return pts / norms

    def compute_facet_data(pts):
        """
        Compute all facet circumcenters and their angular radii (vectorized).
        Returns: (hull, radii_array, centers_array, simplices_array)
        """
        try:
            hull = ConvexHull(pts)
        except Exception:
            return None, np.array([]), np.array([]), np.array([])
        
        simplices = hull.simplices  # shape (F, 3)
        F = len(simplices)
        
        v1 = pts[simplices[:, 0]]
        v2 = pts[simplices[:, 1]]
        v3 = pts[simplices[:, 2]]
        
        edge1 = v2 - v1
        edge2 = v3 - v1
        normals = np.cross(edge1, edge2)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        
        valid = (norms[:, 0] > 1e-12)
        if not np.any(valid):
            return hull, np.array([]), np.array([]), np.array([])
        
        normals = normals[valid]
        norms = norms[valid]
        v1 = v1[valid]
        simplices = simplices[valid]
        
        centers = normals / norms
        dots = np.sum(centers * v1, axis=1)
        centers[dots < 0] *= -1
        
        diffs = centers - v1
        d = np.linalg.norm(diffs, axis=1)
        radii = 2.0 * np.arcsin(np.clip(d / 2.0, 0.0, 1.0))
        
        return hull, radii, centers, simplices

    def compute_covering_radius(pts):
        """Compute the covering radius in radians and the worst-case circumcenter."""
        hull, radii, centers, simplices = compute_facet_data(pts)
        if len(radii) == 0:
            return np.pi, np.array([0.0, 0.0, 1.0])
        
        max_idx = np.argmax(radii)
        return radii[max_idx], centers[max_idx]

    def spectral_energy_and_grad(pts):
        """
        Compute spectral energy (variance of pairwise distances) and its gradient.
        Minimizing this promotes uniform distribution on the sphere.
        """
        n = len(pts)
        dots = np.dot(pts, pts.T)  # (n, n)
        
        mask = ~np.eye(n, dtype=bool)
        mean_dot = np.sum(dots[mask]) / (n * (n - 1))
        
        grad = np.zeros_like(pts)
        
        for k in range(n):
            dots_k = dots[k, :].copy()
            dots_k[k] = 0
            
            terms = dots_k * (dots_k**2 - 1)
            grad_k = np.sum(pts * terms[:, np.newaxis], axis=0) - np.sum(terms * dots_k) * pts[k]
            
            grad_k = grad_k - np.dot(grad_k, pts[k]) * pts[k]
            grad[k] = grad_k
        
        E = 0.5 * np.sum((1 - dots[mask]**2))
        
        return E, grad

    def soft_max_radius_and_grad(pts, beta=50.0):
        """
        Compute soft-max radius and its gradient w.r.t. points using analytical gradients.
        Returns: (soft_radius, gradient_array)
        """
        hull, radii, centers, simplices = compute_facet_data(pts)
        if len(radii) == 0:
            return np.pi, np.zeros_like(pts)
        
        max_r = np.max(radii)
        shifted = beta * (radii - max_r)
        max_shifted = np.max(shifted)
        exp_terms = np.exp(shifted - max_shifted)
        log_sum = max_shifted + np.log(np.sum(exp_terms))
        soft_radius = max_r + log_sum / beta
        weights = exp_terms / np.sum(exp_terms)
        
        grads = np.zeros_like(pts)
        F = len(radii)
        
        for i in range(F):
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
            
            simplex = simplices[i]
            p_pts = pts[simplex]
            
            grad_contrib = -C / sin_r
            
            dots = np.sum(p_pts * grad_contrib, axis=1)
            radial_comps = dots[:, np.newaxis] * p_pts
            tangential_grads = grad_contrib - radial_comps
            
            for k in range(3):
                grads[simplex[k]] += w * tangential_grads[k]
        
        return soft_radius, grads

    def repulsion_relax(pts, time_budget, alpha_start=0.05):
        """
        Repulsion relaxation: minimize sum 1/||p_i - p_j||^2 via gradient descent.
        Helps spread points more uniformly before covering optimization.
        """
        pts = pts.copy()
        alpha = alpha_start
        start_t = time.time()
        
        def repulsion_energy(p):
            diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dist_sq = np.sum(diff * diff, axis=2)
            mask = ~np.eye(N, dtype=bool)
            inv_dist_sq = np.zeros_like(dist_sq)
            np.divide(1.0, dist_sq, out=inv_dist_sq, where=mask)
            return 0.5 * np.sum(inv_dist_sq)
        
        current_energy = repulsion_energy(pts)
        
        while time.time() - start_t < time_budget:
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dist_sq = np.sum(diff * diff, axis=2)
            
            mask = np.eye(N, dtype=bool)
            inv_dist_sq = np.zeros_like(dist_sq)
            np.divide(1.0, dist_sq, out=inv_dist_sq, where=~mask)
            
            forces = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
            
            dot_fp = np.sum(forces * pts, axis=1)
            forces_tan = forces - dot_fp[:, np.newaxis] * pts
            
            step = alpha
            improved = False
            for _ in range(8):
                new_pts = pts + step * forces_tan
                new_pts = project_to_sphere(new_pts)
                new_energy = repulsion_energy(new_pts)
                if new_energy < current_energy:
                    pts = new_pts
                    current_energy = new_energy
                    improved = True
                    break
                step *= 0.5
            
            if not improved:
                alpha *= 0.85
            else:
                alpha = min(alpha * 1.02, 0.08)
            
            alpha = max(alpha, 1e-6)
        
        return pts

    def lloyd_iteration(pts, time_budget, step_size=0.03):
        """
        Weighted Lloyd iteration: move each point toward weighted average of
        circumcenters of facets it belongs to, weighted by facet radius.
        """
        pts = pts.copy()
        start_t = time.time()
        
        while time.time() - start_t < time_budget:
            hull, radii, centers, simplices = compute_facet_data(pts)
            if hull is None or len(radii) == 0:
                break
            
            max_rad = np.max(radii)
            if max_rad < 1e-10:
                break
            
            updates = np.zeros_like(pts)
            weights_sum = np.zeros(N)
            
            for f in range(len(simplices)):
                rad = radii[f]
                if rad < 1e-12:
                    continue
                w = (rad / max_rad) ** 2
                cc = centers[f]
                for idx in simplices[f]:
                    updates[idx] += w * (cc - pts[idx])
                    weights_sum[idx] += w
            
            mask = weights_sum > 1e-10
            if np.any(mask):
                pts[mask] += step_size * updates[mask] / weights_sum[mask, np.newaxis]
                pts = project_to_sphere(pts)
        
        return pts

    def optimize_worst_facet(pts, step_size, iterations=50):
        """
        Iteratively move the vertices of the worst facet towards the worst center.
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

            hull, radii, centers, simplices = compute_facet_data(pts)
            if hull is None or len(radii) == 0:
                break

            target_indices_set = set()
            for i in range(len(simplices)):
                if np.dot(centers[i], worst_center) > 0.999:
                    for idx in simplices[i]:
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

    # Phase 1: Generate multiple diverse initial configurations
    initial_configs = []
    
    initial_configs.append(fibonacci_sphere(N))
    initial_configs.append(fibonacci_sphere(N, offset=0.3))
    
    np.random.seed(123)
    rand_pts = np.random.randn(N, 3)
    rand_pts /= np.linalg.norm(rand_pts, axis=1, keepdims=True)
    initial_configs.append(rand_pts)
    
    np.random.seed(456)
    rand_pts2 = np.random.randn(N, 3)
    rand_pts2 /= np.linalg.norm(rand_pts2, axis=1, keepdims=True)
    initial_configs.append(rand_pts2)
    
    best = None
    best_radius = np.pi
    for config in initial_configs:
        radius, _ = compute_covering_radius(config)
        if radius < best_radius:
            best_radius = radius
            best = config.copy()
    
    current = best.copy()

    # Phase 2: Repulsion relaxation for better initial spread
    if time.time() < deadline - 40:
        try:
            relaxed = repulsion_relax(current, time_budget=10.0)
            radius, _ = compute_covering_radius(relaxed)
            if radius < best_radius:
                best_radius = radius
                best = relaxed.copy()
                current = relaxed.copy()
        except Exception:
            pass

    # Phase 3: Lloyd iteration for structured improvement
    if time.time() < deadline - 35:
        try:
            lloyd_pts = lloyd_iteration(current, time_budget=8.0, step_size=0.05)
            radius, _ = compute_covering_radius(lloyd_pts)
            if radius < best_radius:
                best_radius = radius
                best = lloyd_pts.copy()
                current = lloyd_pts.copy()
        except Exception:
            pass

    # Phase 4: Spectral energy minimization (smooth surrogate)
    if time.time() < deadline - 30:
        try:
            x0 = current.flatten()
            
            def spectral_obj_and_grad(x):
                if time.time() > deadline - 25:
                    return 1e10, np.zeros_like(x)
                pts = x.reshape(N, 3)
                norms = np.linalg.norm(pts, axis=1, keepdims=True)
                norms = np.where(norms < 1e-10, 1.0, norms)
                pts = pts / norms
                energy, grad = spectral_energy_and_grad(pts)
                return energy, grad.reshape(-1)
            
            elapsed = time.time() - start_time
            remaining = deadline - 30 - elapsed
            if remaining > 5:
                max_iter = int(min(800, max(100, 200 * (remaining / 10.0))))
                
                result = minimize(
                    spectral_obj_and_grad,
                    x0,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': max_iter, 'maxfun': max_iter*3, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 200}
                )
                
                if result.success or result.fun < spectral_obj_and_grad(x0)[0]:
                    spectral_pts = project_to_sphere(result.x.reshape(N, 3))
                    spectral_radius, _ = compute_covering_radius(spectral_pts)
                    if spectral_radius < best_radius:
                        best_radius = spectral_radius
                        best = spectral_pts.copy()
                        current = spectral_pts.copy()
        except Exception:
            pass

    # Phase 5: L-BFGS-B with soft-max covering radius (progressive beta)
    if time.time() < deadline - 25:
        try:
            x0 = current.flatten()
            beta_schedule = [50.0, 100.0, 200.0, 400.0, 800.0]
            for beta in beta_schedule:
                if time.time() > deadline - 25:
                    break
                
                def objective_and_grad(x, b=beta):
                    if time.time() > deadline - 20:
                        return 1e10, np.zeros_like(x)
                    pts = x.reshape(N, 3)
                    norms = np.linalg.norm(pts, axis=1, keepdims=True)
                    norms = np.where(norms < 1e-10, 1.0, norms)
                    pts = pts / norms
                    soft_r, grads = soft_max_radius_and_grad(pts, beta=b)
                    return soft_r, grads.reshape(-1)
                
                elapsed = time.time() - start_time
                remaining = deadline - 25 - elapsed
                if remaining < 3:
                    break
                max_iter = int(min(400, max(50, 100 * (remaining / 5.0))))
                
                result = minimize(
                    objective_and_grad,
                    x0,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': max_iter, 'maxfun': max_iter*3, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 200}
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

    # Phase 6: Worst-facet optimization with adaptive step size
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

    # Phase 7: Random restarts with spectral energy refinement
    if time.time() < deadline - 20:
        np.random.seed(42)
        for restart in range(50):
            if time.time() > deadline - 20:
                break
            noise_scale = 0.05 * (0.7 ** (restart // 10))
            perturbed = best.copy()
            for i in range(N):
                rand_vec = np.random.randn(3)
                rand_vec /= np.linalg.norm(rand_vec)
                perturbed[i] = perturbed[i] + noise_scale * rand_vec
                perturbed[i] /= np.linalg.norm(perturbed[i])
            
            if time.time() < deadline - 20:
                try:
                    x0 = perturbed.flatten()
                    
                    def quick_spectral_obj_and_grad(x):
                        if time.time() > deadline - 20:
                            return 1e10, np.zeros_like(x)
                        pts = x.reshape(N, 3)
                        norms = np.linalg.norm(pts, axis=1, keepdims=True)
                        norms = np.where(norms < 1e-10, 1.0, norms)
                        pts = pts / norms
                        energy, grad = spectral_energy_and_grad(pts)
                        return energy, grad.reshape(-1)
                    
                    result = minimize(
                        quick_spectral_obj_and_grad,
                        x0,
                        method='L-BFGS-B',
                        jac=True,
                        options={'maxiter': 80, 'maxfun': 240, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 100}
                    )
                    
                    if result.success:
                        refined_pts = project_to_sphere(result.x.reshape(N, 3))
                        radius, _ = compute_covering_radius(refined_pts)
                        if radius < best_radius:
                            best_radius = radius
                            best = refined_pts.copy()
                            current = refined_pts.copy()
                except Exception:
                    pass
            
            for step in [0.02, 0.01, 0.005, 0.002, 0.001]:
                if time.time() > deadline - 20:
                    break
                perturbed = optimize_worst_facet(perturbed, step, iterations=60)
            radius, _ = compute_covering_radius(perturbed)
            if radius < best_radius:
                best_radius = radius
                best = perturbed.copy()

    # Phase 8: Final high-precision polish
    if time.time() < deadline - 5:
        final = best.copy()
        beta_final = 1000.0
        
        def final_objective_and_grad(x, b=beta_final):
            if time.time() > deadline - 3:
                return 1e10, np.zeros_like(x)
            pts = x.reshape(N, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms = np.where(norms < 1e-10, 1.0, norms)
            pts = pts / norms
            soft_r, grads = soft_max_radius_and_grad(pts, beta=b)
            return soft_r, grads.reshape(-1)
        
        try:
            x0 = final.flatten()
            remaining = deadline - 5 - (time.time() - start_time)
            if remaining > 3:
                max_iter = int(min(600, max(100, 200 * (remaining / 5.0))))
                result = minimize(
                    final_objective_and_grad,
                    x0,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': max_iter, 'maxfun': max_iter*3, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 300}
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
        if time.time() < deadline - 2:
            final = best.copy()
            step = 0.001
            prev_radius = compute_covering_radius(final)[0]
            stall_count = 0
            for it in range(3000):
                if time.time() > deadline - 2:
                    break
                radius, worst_center = compute_covering_radius(final)
                if radius >= prev_radius - 1e-16:
                    step *= 0.7
                    stall_count += 1
                    if step < 1e-16 or stall_count > 100:
                        break
                else:
                    prev_radius = radius
                    stall_count = 0

                hull, radii, centers, simplices = compute_facet_data(final)
                if hull is None or len(radii) == 0:
                    break

                target_indices_set = set()
                for i in range(len(simplices)):
                    if np.dot(centers[i], worst_center) > 0.99999:
                        for idx in simplices[i]:
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
