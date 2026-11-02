import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s kill

    def fibonacci_initial():
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return pts / norms

    def compute_hull_data(p):
        """Vectorized circumcenter/radius computation via cross-product."""
        try:
            hull = ConvexHull(p)
            facets = hull.simplices
        except Exception:
            return None, None, None

        n_facets = len(facets)
        if n_facets == 0:
            return None, None, None

        a = p[facets[:, 0]]
        b = p[facets[:, 1]]
        c = p[facets[:, 2]]

        v1 = a - b
        v2 = b - c
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n, axis=1)

        circumcenters = np.zeros((n_facets, 3))
        radii = np.zeros(n_facets)

        valid = norm_n > 1e-12
        if np.any(valid):
            cc = n[valid] / norm_n[valid, np.newaxis]
            # Ensure circumcenter is on the correct side (outward, same hemisphere as facet)
            dots = np.sum(cc * a[valid], axis=1)
            flip = dots < 0
            if np.any(flip):
                cc[flip] *= -1
            circumcenters[valid] = cc

            cos_r = np.clip(np.sum(cc * a[valid], axis=1), -1.0, 1.0)
            radii[valid] = np.arccos(cos_r)

        return facets, circumcenters, radii

    def soft_max_objective_and_grad(pts, beta):
        """Soft-max covering radius via log-sum-exp, with analytic gradient."""
        facets, circumcenters, radii = compute_hull_data(pts)
        if facets is None:
            return np.pi, np.zeros_like(pts)

        n_facets = len(facets)
        if n_facets == 0:
            return np.pi, np.zeros_like(pts)

        max_r = np.max(radii)
        shifted = beta * (radii - max_r)
        max_shifted = np.max(shifted)
        exp_terms = np.exp(shifted - max_shifted)
        sum_exp = np.sum(exp_terms)
        log_sum = max_shifted + np.log(sum_exp)
        soft_radius = max_r + log_sum / beta
        weights = exp_terms / sum_exp

        grads = np.zeros_like(pts)

        for i in range(n_facets):
            w = weights[i]
            if w < 1e-15:
                continue
            r = radii[i]
            if r < 1e-12:
                continue

            C = circumcenters[i]
            sin_r = np.sin(r)
            if sin_r < 1e-10:
                continue

            simplex = facets[i]
            for k in range(3):
                p_idx = simplex[k]
                p = pts[p_idx]
                # Gradient of arccos(dot(C, p)) w.r.t. p is -C/sin(r)
                # Project to tangent space at p
                grad_contrib = -C / sin_r
                radial = np.dot(grad_contrib, p) * p
                tangential = grad_contrib - radial
                grads[p_idx] += w * tangential

        return soft_radius, grads

    def sum_sq_objective_and_grad(pts, lam):
        """
        Epigraph-style smooth surrogate: minimize max(r) + lam * sum((r_i - max_r)_+^2).
        Gradient is computed analytically per facet.
        """
        facets, circumcenters, radii = compute_hull_data(pts)
        if facets is None:
            return np.pi, np.zeros_like(pts)

        n_facets = len(facets)
        if n_facets == 0:
            return np.pi, np.zeros_like(pts)

        max_r = np.max(radii)
        # Penalty: facets above max_r (only the worst ones contribute)
        excess = np.maximum(radii - max_r, 0.0)
        penalty = np.sum(excess * excess)
        obj = max_r + lam * penalty

        # Gradient: for the worst facet(s), use soft-max weighting; plus penalty gradient
        # For the penalty term: d/dp_i [sum_j (r_j - max_r)^2 * 1_{r_j > max_r}]
        # This is complex; use soft-max gradient for the max part and add penalty gradient
        # Simplified: use soft-max with high beta for the max part
        beta_eff = 200.0
        shifted = beta_eff * (radii - max_r)
        max_shifted = np.max(shifted)
        exp_terms = np.exp(shifted - max_shifted)
        sum_exp = np.sum(exp_terms)
        weights = exp_terms / sum_exp

        grads = np.zeros_like(pts)
        for i in range(n_facets):
            w = weights[i]
            if w < 1e-15:
                continue
            r = radii[i]
            if r < 1e-12:
                continue
            C = circumcenters[i]
            sin_r = np.sin(r)
            if sin_r < 1e-10:
                continue
            simplex = facets[i]
            for k in range(3):
                p_idx = simplex[k]
                p = pts[p_idx]
                grad_contrib = -C / sin_r
                radial = np.dot(grad_contrib, p) * p
                tangential = grad_contrib - radial
                grads[p_idx] += w * tangential

        return obj, grads

    def compute_forces(pts, beta=50.0):
        """Compute smooth forces from soft-max energy + repulsion."""
        facets, circumcenters, radii = compute_hull_data(pts)
        if facets is None:
            return None, None, None, None

        n_facets = len(facets)
        if n_facets == 0:
            return None, None, None, None

        r2 = radii * radii
        max_r2 = np.max(r2)
        exp_terms = np.exp(beta * (r2 - max_r2))
        sum_exp = np.sum(exp_terms)
        weights = (2.0 / beta) * radii * exp_terms / sum_exp

        forces = np.zeros_like(pts)

        for k in range(n_facets):
            w = weights[k]
            if w < 1e-15:
                continue
            cc = circumcenters[k]
            for idx in facets[k]:
                forces[idx] += w * (cc - pts[idx])

        # Repulsive forces (vectorized)
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diffs * diffs, axis=2)
        np.fill_diagonal(dist_sq, 1.0)

        inv_dist_sq = 1.0 / dist_sq
        np.fill_diagonal(inv_dist_sq, 0.0)

        forces_rep = np.sum(diffs * inv_dist_sq[:, :, np.newaxis], axis=1)
        forces += 0.1 * forces_rep

        return forces, facets, circumcenters, radii

    def optimize_worst_facet(pts, step_size, iterations=100):
        """Iteratively move vertices of worst facet toward its circumcenter."""
        pts = pts.copy()
        prev_radius = np.pi

        for _ in range(iterations):
            if time.time() > deadline - 5:
                break

            facets, circumcenters, radii = compute_hull_data(pts)
            if facets is None:
                break

            current_radius = np.max(radii)
            worst_idx = np.argmax(radii)
            worst_center = circumcenters[worst_idx]

            if current_radius >= prev_radius:
                step_size *= 0.5
                if step_size < 1e-8:
                    break
            else:
                prev_radius = current_radius

            target_indices_set = set()
            for k in range(len(facets)):
                if np.dot(circumcenters[k], worst_center) > 0.999:
                    for idx in facets[k]:
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

    def lloyd_iteration(pts, time_budget, step_size=0.03):
        """Weighted Lloyd iteration."""
        pts = pts.copy()
        start_t = time.time()

        while time.time() - start_t < time_budget:
            facets, circumcenters, radii = compute_hull_data(pts)
            if facets is None:
                break

            max_rad = np.max(radii)
            if max_rad < 1e-10:
                break

            updates = np.zeros_like(pts)
            weights_sum = np.zeros(N)

            for f in range(len(facets)):
                rad = radii[f]
                if rad < 1e-12:
                    continue
                w = (rad / max_rad) ** 2
                cc = circumcenters[f]
                for idx in facets[f]:
                    updates[idx] += w * (cc - pts[idx])
                    weights_sum[idx] += w

            mask = weights_sum > 1e-10
            if np.any(mask):
                pts[mask] += step_size * updates[mask] / weights_sum[mask, np.newaxis]
                pts = normalize(pts)

        return pts

    best_pts = fibonacci_initial()
    best_radius = float('inf')

    # Phase 1: Coarse optimization with soft-max forces
    pts = best_pts.copy()
    eta = 0.02
    beta = 30.0

    for step in range(1500):
        if time.time() > deadline - 40.0:
            break

        forces, facets, circumcenters, radii = compute_forces(pts, beta)
        if forces is None:
            break

        pts += eta * forces
        pts = normalize(pts)

        if radii is not None:
            current_radius = np.max(radii)
            if current_radius < best_radius:
                best_radius = current_radius
                best_pts = pts.copy()

        eta *= 0.998
        beta += 0.5

    # Phase 2: L-BFGS-B on soft-max radius with analytic gradient
    if time.time() < deadline - 30.0:
        pts = best_pts.copy()

        for beta_val in [80.0, 150.0, 300.0]:
            if time.time() > deadline - 15.0:
                break

            def objective_and_grad(params):
                X = normalize(params.reshape(N, 3))
                soft_r, grads = soft_max_objective_and_grad(X, beta_val)
                return soft_r, grads.flatten()

            try:
                res = minimize(objective_and_grad, pts.flatten(), method='L-BFGS-B',
                               jac=True,
                               options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100})

                new_pts = normalize(res.x.reshape(N, 3))
                facets, circumcenters, radii = compute_hull_data(new_pts)
                if radii is not None:
                    current_radius = np.max(radii)
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_pts = new_pts.copy()
                        pts = new_pts.copy()
            except Exception:
                pass

    # Phase 2b: L-BFGS-B on sum-of-squares epigraph objective
    if time.time() < deadline - 20.0:
        pts = best_pts.copy()
        for lam_val in [5.0, 20.0]:
            if time.time() > deadline - 10.0:
                break

            def sum_sq_obj_and_grad(params):
                X = normalize(params.reshape(N, 3))
                obj, grads = sum_sq_objective_and_grad(X, lam_val)
                return obj, grads.flatten()

            try:
                res = minimize(sum_sq_obj_and_grad, pts.flatten(), method='L-BFGS-B',
                               jac=True,
                               options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100})

                new_pts = normalize(res.x.reshape(N, 3))
                facets, circumcenters, radii = compute_hull_data(new_pts)
                if radii is not None:
                    current_radius = np.max(radii)
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_pts = new_pts.copy()
                        pts = new_pts.copy()
            except Exception:
                pass

    # Phase 3: Lloyd iteration for structural refinement
    if time.time() < deadline - 20.0:
        time_budget = min(10.0, deadline - time.time() - 10.0)
        if time_budget > 2.0:
            lloyd_pts = lloyd_iteration(best_pts.copy(), time_budget, step_size=0.03)
            facets, circumcenters, radii = compute_hull_data(lloyd_pts)
            if radii is not None:
                radius = np.max(radii)
                if radius < best_radius:
                    best_radius = radius
                    best_pts = lloyd_pts.copy()

    # Phase 4: Adaptive worst-facet optimization with multiple step sizes
    if time.time() < deadline - 15.0:
        step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
        for step in step_sizes:
            if time.time() > deadline - 10:
                break

            current = optimize_worst_facet(best_pts.copy(), step, iterations=80)
            facets, circumcenters, radii = compute_hull_data(current)
            if radii is not None:
                radius = np.max(radii)
                if radius < best_radius:
                    best_radius = radius
                    best_pts = current.copy()

    # Phase 5: Random restarts with small perturbations
    if time.time() < deadline - 10.0:
        rng = np.random.RandomState(42)
        for restart in range(30):
            if time.time() > deadline - 5:
                break

            noise = rng.randn(N, 3) * 0.05
            perturbed = best_pts + noise
            perturbed = normalize(perturbed)

            pts = perturbed.copy()
            eta = 0.01
            beta = 80.0

            for step in range(300):
                if time.time() > deadline - 3:
                    break

                forces, facets, circumcenters, radii = compute_forces(pts, beta)
                if forces is None:
                    break

                pts += eta * forces
                pts = normalize(pts)

                if radii is not None:
                    current_radius = np.max(radii)
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_pts = pts.copy()

                eta *= 0.995
                beta += 0.3

            if time.time() < deadline - 3:
                current = optimize_worst_facet(pts, 0.01, iterations=50)
                facets, circumcenters, radii = compute_hull_data(current)
                if radii is not None:
                    radius = np.max(radii)
                    if radius < best_radius:
                        best_radius = radius
                        best_pts = current.copy()

    # Phase 6: Final fine polish with high-beta soft-max gradient
    if time.time() < deadline - 1.0:
        pts = best_pts.copy()

        for beta_val in [300.0, 500.0, 800.0]:
            if time.time() > deadline - 0.5:
                break

            def final_obj_and_grad(params):
                X = normalize(params.reshape(N, 3))
                soft_r, grads = soft_max_objective_and_grad(X, beta_val)
                return soft_r, grads.flatten()

            try:
                res = minimize(final_obj_and_grad, pts.flatten(), method='L-BFGS-B',
                               jac=True,
                               options={'maxiter': 100, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50})
                new_pts = normalize(res.x.reshape(N, 3))
                facets, circumcenters, radii = compute_hull_data(new_pts)
                if radii is not None:
                    current_radius = np.max(radii)
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_pts = new_pts.copy()
                        pts = new_pts.copy()
            except Exception:
                pass

        # Very fine gradient steps
        eta = 0.001
        beta = 500.0
        for step in range(100):
            if time.time() > deadline - 0.5:
                break

            forces, facets, circumcenters, radii = compute_forces(pts, beta)
            if forces is None:
                break

            pts += eta * forces
            pts = normalize(pts)

            if radii is not None:
                current_radius = np.max(radii)
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_pts = pts.copy()

            eta *= 0.995

    # Final validation
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-10) or not np.all(np.isfinite(best_pts)):
        best_pts = fibonacci_initial()

    best_pts = normalize(best_pts)

    return best_pts
