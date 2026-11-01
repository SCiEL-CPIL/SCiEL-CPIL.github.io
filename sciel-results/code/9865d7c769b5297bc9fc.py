"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    def fibonacci_sphere(n, offset=0.5):
        i = np.arange(n) + offset
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        return pts

    def project_to_sphere(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        return pts / norms

    def compute_facet_data(pts):
        """
        Compute all facet circumcenters and their angular radii (vectorized).
        Returns: (radii_array, centers_array, simplices_array)
        """
        try:
            hull = ConvexHull(pts)
        except Exception:
            return np.array([]), np.array([]), np.array([])

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
            return np.array([]), np.array([]), np.array([])

        normals = normals[valid]
        norms = norms[valid]
        v1 = v1[valid]
        v2 = v2[valid]
        v3 = v3[valid]
        simplices = simplices[valid]

        centers = normals / norms

        dots = np.sum(centers * v1, axis=1)
        centers[dots < 0] *= -1

        diffs = centers - v1
        d = np.linalg.norm(diffs, axis=1)
        radii = 2.0 * np.arcsin(np.clip(d / 2.0, 0.0, 1.0))

        return radii, centers, simplices

    def compute_covering_radius(pts):
        """
        Compute the covering radius in radians and the worst-case circumcenter.
        """
        radii, centers, simplices = compute_facet_data(pts)
        if len(radii) == 0:
            return np.pi, np.array([0.0, 0.0, 1.0])

        max_idx = np.argmax(radii)
        return radii[max_idx], centers[max_idx]

    def soft_max_radius_and_grad(pts, beta=50.0):
        """
        Compute soft-max radius and its gradient w.r.t. points using analytical gradients.
        Returns: (soft_radius, gradient_array)
        """
        radii, centers, simplices = compute_facet_data(pts)
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

    def repulsion_relax(pts, iterations=200, k_repulse=0.5, k_attract=0.5):
        """
        Repulsion-based relaxation: points repel each other to spread out.
        This is a different paradigm from spring-mass on hull edges.
        """
        pts = pts.copy()
        damping = 0.1

        for it in range(iterations):
            if time.time() > deadline - 20:
                break

            forces = np.zeros_like(pts)
            for i in range(N):
                for j in range(i + 1, N):
                    v_i = pts[i]
                    v_j = pts[j]
                    dot = np.dot(v_i, v_j)
                    dot = np.clip(dot, -1.0, 1.0)
                    # Repulsive force: stronger when closer
                    force_mag = k_repulse * (1.0 - dot) ** 2

                    # Direction: from j to i in tangent space at i
                    dir_vec = v_i - dot * v_j
                    dir_norm = np.linalg.norm(dir_vec)
                    if dir_norm > 1e-10:
                        dir_vec /= dir_norm
                        forces[i] += force_mag * dir_vec
                        forces[j] -= force_mag * dir_vec

            for i in range(N):
                v = pts[i]
                f_tangent = forces[i] - np.dot(forces[i], v) * v
                pts[i] = v + damping * f_tangent
                pts[i] /= np.linalg.norm(pts[i])

        return pts

    def lloyd_iteration(pts, iterations=100):
        """
        Lloyd's algorithm: iteratively move each point to the centroid of its
        Voronoi cell (intersection of hemispheres). This is a different
        optimization paradigm from gradient descent.
        """
        pts = pts.copy()

        for it in range(iterations):
            if time.time() > deadline - 20:
                break

            # For each point, find its Voronoi cell centroid
            # Approximate by finding points closer to it than any other
            new_pts = pts.copy()
            for i in range(N):
                # Find all points closer to i than to any other point
                # This is an O(N^2) approximation
                closer = []
                for j in range(N):
                    if i == j:
                        continue
                    # Check if j is closer to i than to any other point
                    is_closer = True
                    for k in range(N):
                        if k == i or k == j:
                            continue
                        if np.dot(pts[j], pts[k]) > np.dot(pts[j], pts[i]):
                            is_closer = False
                            break
                    if is_closer:
                        closer.append(j)

                if closer:
                    # Move toward the centroid of closer points
                    centroid = np.mean(pts[closer], axis=0)
                    centroid /= np.linalg.norm(centroid)
                    new_pts[i] = 0.5 * pts[i] + 0.5 * centroid
                    new_pts[i] /= np.linalg.norm(new_pts[i])

            pts = project_to_sphere(new_pts)

        return pts

    def facet_targeted_move(pts, step_size, iterations=50):
        """
        Directly move vertices of the worst facets toward their circumcenters.
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

    # Phase 1: Initial Fibonacci spiral
    current = fibonacci_sphere(N)
    best = current.copy()
    best_radius, _ = compute_covering_radius(best)

    # Phase 2: Repulsion relaxation (different from spring-mass)
    if time.time() < deadline - 40:
        try:
            repel_pts = repulsion_relax(current, iterations=200)
            repel_radius, _ = compute_covering_radius(repel_pts)
            if repel_radius < best_radius:
                best_radius = repel_radius
                best = repel_pts.copy()
                current = repel_pts.copy()
        except Exception:
            pass

    # Phase 3: Lloyd's iteration (different paradigm)
    if time.time() < deadline - 30:
        try:
            lloyd_pts = lloyd_iteration(current, iterations=100)
            lloyd_radius, _ = compute_covering_radius(lloyd_pts)
            if lloyd_radius < best_radius:
                best_radius = lloyd_radius
                best = lloyd_pts.copy()
                current = lloyd_pts.copy()
        except Exception:
            pass

    # Phase 4: L-BFGS-B with analytical gradients
    if time.time() < deadline - 25:
        try:
            x0 = current.flatten()
            beta_schedule = [30.0, 80.0, 150.0, 300.0]
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
                max_iter = int(min(500, max(100, 150 * (remaining / 30.0))))

                result = minimize(
                    objective_and_grad,
                    x0,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': max_iter, 'maxfun': max_iter * 3, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 200}
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

    # Phase 5: Facet-targeted moves with adaptive step size
    step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    for step in step_sizes:
        if time.time() > deadline - 15:
            break
        current = facet_targeted_move(current, step, iterations=100)
        radius, _ = compute_covering_radius(current)
        if radius < best_radius:
            best_radius = radius
            best = current.copy()
        else:
            current = best.copy()

    # Phase 6: Random restarts with varied perturbations
    if time.time() < deadline - 25:
        np.random.seed(42)
        for restart in range(80):
            if time.time() > deadline - 25:
                break
            noise_scale = 0.05 * (0.7 ** (restart // 10))
            perturbed = best.copy()
            for i in range(N):
                rand_vec = np.random.randn(3)
                rand_vec /= np.linalg.norm(rand_vec)
                perturbed[i] = perturbed[i] + noise_scale * rand_vec
                perturbed[i] /= np.linalg.norm(perturbed[i])

            for step in [0.02, 0.01, 0.005, 0.002, 0.001]:
                if time.time() > deadline - 25:
                    break
                perturbed = facet_targeted_move(perturbed, step, iterations=80)
            radius, _ = compute_covering_radius(perturbed)
            if radius < best_radius:
                best_radius = radius
                best = perturbed.copy()

    # Phase 7: Final high-precision polish
    if time.time() < deadline - 3:
        final = best.copy()
        beta_final = 800.0

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
                max_iter = int(min(800, max(200, 250 * (remaining / 10.0))))
                result = minimize(
                    final_objective_and_grad,
                    x0,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': max_iter, 'maxfun': max_iter * 3, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 300}
                )
                if result.success:
                    polish_pts = project_to_sphere(result.x.reshape(N, 3))
                    polish_radius, _ = compute_covering_radius(polish_pts)
                    if polish_radius < best_radius:
                        best_radius = polish_radius
                        best = polish_pts.copy()
        except Exception:
            pass

        if time.time() < deadline - 1:
            final = best.copy()
            step = 0.001
            prev_radius = compute_covering_radius(final)[0]
            stall_count = 0
            for it in range(5000):
                if time.time() > deadline - 1:
                    break
                radius, worst_center = compute_covering_radius(final)
                if radius >= prev_radius - 1e-16:
                    step *= 0.7
                    stall_count += 1
                    if step < 1e-16 or stall_count > 150:
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
