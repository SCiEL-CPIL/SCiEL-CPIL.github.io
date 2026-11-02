"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

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

    def compute_facet_data(points):
        """
        Compute convex hull and facet circumcenters/radii (vectorized).
        Returns: hull, facets, centers (array), radii (array), worst_idx
        """
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        if np.any(norms < 1e-15):
            return None, None, None, None, None
        points = points / norms
        
        try:
            hull = ConvexHull(points)
        except Exception:
            return None, None, None, None, None
        
        facets = hull.simplices
        n_facets = len(facets)
        if n_facets == 0:
            return hull, facets, np.zeros((0,3)), np.zeros(0), None
            
        a = points[facets[:, 0]]
        b = points[facets[:, 1]]
        c = points[facets[:, 2]]
        
        v1 = a - b
        v2 = b - c
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n, axis=1)
        
        valid = norm_n > 1e-12
        
        centers = np.zeros((n_facets, 3))
        radii = np.zeros(n_facets)
        
        if np.any(valid):
            cc = n[valid] / norm_n[valid, np.newaxis]
            
            dots = np.sum(cc * a[valid], axis=1)
            cc[dots < 0] *= -1
            
            centers[valid] = cc
            
            cos_r = np.clip(np.sum(cc * a[valid], axis=1), -1.0, 1.0)
            radii[valid] = np.arccos(cos_r)
        
        if n_facets > 0 and np.any(radii > 0):
            worst_idx = np.argmax(radii)
        else:
            worst_idx = None
            
        return hull, facets, centers, radii, worst_idx

    def compute_covering_radius(points):
        hull, facets, centers, radii, worst_idx = compute_facet_data(points)
        if hull is None or worst_idx is None:
            return np.pi, np.array([0.0, 0.0, 1.0])
        return radii[worst_idx], centers[worst_idx]

    def soft_max_radius_objective(flat_pts, beta=80.0):
        """
        Soft-max objective: smooth approximation of max facet radius.
        Uses log-sum-exp for numerical stability.
        """
        pts = flat_pts.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        pts = pts / norms

        hull, facets, centers, radii, worst_idx = compute_facet_data(pts)
        if hull is None or len(radii) < 2:
            return np.pi

        max_r = np.max(radii)
        if max_r < 1e-15:
            return 0.0
            
        shifted = beta * (radii - max_r)
        max_shifted = np.max(shifted)
        exp_terms = np.exp(shifted - max_shifted)
        log_sum = max_shifted + np.log(np.sum(exp_terms))
        soft_radius = max_r + log_sum / beta
        return soft_radius

    def epigraph_objective(flat_pts, beta=50.0):
        """
        Epigraph formulation: minimize t subject to all radii <= t.
        Uses a smooth approximation: t + (1/beta) * sum(exp(beta*(r_i - t)))
        This is smoother than soft-max for L-BFGS-B.
        """
        pts = flat_pts.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        pts = pts / norms

        hull, facets, centers, radii, worst_idx = compute_facet_data(pts)
        if hull is None or len(radii) < 2:
            return np.pi

        t = np.max(radii)
        if t < 1e-15:
            return 0.0

        # Smooth epigraph: t + (1/beta) * sum(exp(beta*(r_i - t)))
        # The sum term is always >= 0, and equals log(n) when all r_i = t
        shifted = beta * (radii - t)
        max_shifted = np.max(shifted)
        exp_terms = np.exp(shifted - max_shifted)
        log_sum = max_shifted + np.log(np.sum(exp_terms))
        return t + log_sum / beta

    def optimize_worst_facet(pts, step_size, iterations=50):
        """
        Directly move vertices of the worst facet toward its circumcenter.
        """
        pts = pts.copy()
        for _ in range(iterations):
            if time.time() > deadline - 5:
                break

            radius, worst_center = compute_covering_radius(pts)
            if radius < 1e-10:
                break

            try:
                hull, facets, centers, radii, worst_idx = compute_facet_data(pts)
                if hull is None:
                    break
            except:
                break

            # Find the facet corresponding to the worst center
            target_indices = None
            for f in range(len(facets)):
                if np.dot(centers[f], worst_center) > 0.9999:
                    target_indices = facets[f]
                    break

            if target_indices is None:
                # Fallback: find facet with max radius
                if worst_idx is not None:
                    target_indices = facets[worst_idx]
                else:
                    break

            for idx in target_indices:
                v = pts[idx]
                dir_vec = worst_center - v
                norm_dir = np.linalg.norm(dir_vec)
                if norm_dir > 1e-10:
                    dir_vec /= norm_dir
                    pts[idx] = v + step_size * dir_vec
                    pts[idx] /= np.linalg.norm(pts[idx])

        return pts

    def lloyd_iteration(pts, iterations=30):
        """
        Lloyd iteration: for each facet, move its vertices toward the facet's
        circumcenter weighted by facet size.
        """
        pts = pts.copy()
        for _ in range(iterations):
            if time.time() > deadline - 5:
                break

            hull, facets, centers, radii, worst_idx = compute_facet_data(pts)
            if hull is None:
                break

            max_rad = np.max(radii)
            if max_rad < 1e-10:
                break

            displacement = np.zeros((N, 3))
            count = np.zeros(N)
            for f in range(len(facets)):
                if radii[f] < 1e-12:
                    continue
                # Use linear weighting for gentler bias toward larger facets
                weight = radii[f] / max_rad
                center = centers[f]
                for idx in facets[f]:
                    displacement[idx] += weight * (center - pts[idx])
                    count[idx] += weight

            mask = count > 1e-10
            if np.any(mask):
                # Use adaptive step size: larger when radii are more uniform
                step = 0.15
                pts[mask] = pts[mask] + step * (displacement[mask] / count[mask, np.newaxis])
                pts = project_to_sphere(pts)

        return pts

    def repulsion_relax(pts, iterations=50, strength=0.01):
        """
        Relax points using pairwise repulsion to spread them more uniformly.
        Vectorized for speed.
        """
        pts = pts.copy()
        for _ in range(iterations):
            if time.time() > deadline - 5:
                break

            # Compute all pairwise dot products
            dots = np.clip(pts @ pts.T, -1.0, 1.0)
            angles = np.arccos(dots)
            
            # Set diagonal to a small value to avoid division by zero
            np.fill_diagonal(angles, 1e-6)
            
            # Force magnitude: strength / d^2
            force_mag = strength / (angles * angles)
            np.fill_diagonal(force_mag, 0.0)
            
            # Direction vectors: pts[i] - pts[j]
            # For each pair (i,j), force on i is in direction pts[i] - pts[j]
            # We can compute this as: for each i, sum over j of force_mag[i,j] * (pts[i] - pts[j])
            # = pts[i] * sum_j force_mag[i,j] - sum_j force_mag[i,j] * pts[j]
            
            # sum_j force_mag[i,j] for each i
            row_sums = force_mag.sum(axis=1, keepdims=True)
            
            # sum_j force_mag[i,j] * pts[j] for each i
            weighted_pts = force_mag @ pts  # shape (N, 3)
            
            forces = pts * row_sums - weighted_pts
            
            # Normalize and project
            for i in range(N):
                new_v = pts[i] + 0.05 * forces[i]
                new_norm = np.linalg.norm(new_v)
                if new_norm > 1e-10:
                    pts[i] = new_v / new_norm

        return pts

    # Phase 1: Initial Fibonacci spiral
    best = fibonacci_sphere(N, 0.5)
    best_radius = compute_covering_radius(best)[0]

    # Phase 2: Pre-processing with Lloyd and repulsion
    current = best.copy()
    if time.time() < deadline - 20:
        current = lloyd_iteration(current, iterations=30)
        radius, _ = compute_covering_radius(current)
        if radius < best_radius:
            best_radius = radius
            best = current.copy()

    if time.time() < deadline - 20:
        current = repulsion_relax(best.copy(), iterations=50)
        radius, _ = compute_covering_radius(current)
        if radius < best_radius:
            best_radius = radius
            best = current.copy()

    # Phase 3: L-BFGS-B optimization on soft-max surrogate
    if time.time() < deadline - 25:
        flat_initial = best.flatten()
        try:
            result = minimize(
                soft_max_radius_objective,
                flat_initial,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if result.success:
                current = result.x.reshape(N, 3)
                radius, _ = compute_covering_radius(current)
                if radius < best_radius:
                    best_radius = radius
                    best = current.copy()
        except:
            pass

    # Phase 3b: Epigraph-based optimization (smoother)
    if time.time() < deadline - 20:
        flat_initial = best.flatten()
        try:
            result = minimize(
                epigraph_objective,
                flat_initial,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if result.success:
                current = result.x.reshape(N, 3)
                radius, _ = compute_covering_radius(current)
                if radius < best_radius:
                    best_radius = radius
                    best = current.copy()
        except:
            pass

    # Phase 4: Worst-facet optimization
    if time.time() < deadline - 15:
        for step in [0.05, 0.02, 0.01]:
            if time.time() > deadline - 15:
                break
            current = optimize_worst_facet(best.copy(), step, iterations=50)
            radius, _ = compute_covering_radius(current)
            if radius < best_radius:
                best_radius = radius
                best = current.copy()

    # Phase 5: Random restarts with perturbations
    if time.time() < deadline - 20:
        for restart in range(12):
            if time.time() > deadline - 20:
                break

            perturbed = best.copy()
            noise_scale = 0.04 / (restart + 1)
            for i in range(N):
                rand_vec = np.random.randn(3)
                rand_vec /= np.linalg.norm(rand_vec)
                perturbed[i] = perturbed[i] + noise_scale * rand_vec
                perturbed[i] /= np.linalg.norm(perturbed[i])

            # Apply Lloyd then L-BFGS
            if time.time() < deadline - 20:
                perturbed = lloyd_iteration(perturbed, iterations=20)

            if time.time() < deadline - 20:
                flat_pert = perturbed.flatten()
                try:
                    result = minimize(
                        soft_max_radius_objective,
                        flat_pert,
                        method='L-BFGS-B',
                        options={'maxiter': 100, 'ftol': 1e-9, 'gtol': 1e-7}
                    )
                    if result.success:
                        perturbed = result.x.reshape(N, 3)
                except:
                    pass

            if time.time() < deadline - 20:
                perturbed = optimize_worst_facet(perturbed, 0.01, iterations=30)

            radius, _ = compute_covering_radius(perturbed)
            if radius < best_radius:
                best_radius = radius
                best = perturbed.copy()

    # Phase 6: Final fine-tuning with simulated annealing
    if time.time() < deadline - 5:
        final = best.copy()
        current_radius = best_radius
        T = 0.01  # Initial temperature
        T_min = 0.001
        cooling = 0.95

        for iteration in range(200):
            if time.time() > deadline - 5:
                break
            if T < T_min:
                break

            # Generate perturbation
            perturbed = final.copy()
            # Move 1-3 random points
            n_move = np.random.randint(1, 4)
            move_indices = np.random.choice(N, size=n_move, replace=False)
            for idx in move_indices:
                noise = np.random.randn(3) * T
                perturbed[idx] = perturbed[idx] + noise
                perturbed[idx] /= np.linalg.norm(perturbed[idx])

            # Evaluate
            new_radius, _ = compute_covering_radius(perturbed)

            # Accept if better or with probability exp(-(delta)/T)
            if new_radius < current_radius:
                final = perturbed
                current_radius = new_radius
            else:
                delta = new_radius - current_radius
                if delta < 0.1:  # Only consider small worsenings
                    prob = np.exp(-delta / T)
                    if np.random.rand() < prob:
                        final = perturbed
                        current_radius = new_radius

            T *= cooling

        radius, _ = compute_covering_radius(final)
        if radius < best_radius:
            best = final
            best_radius = radius

    # Final normalization and validation
    best = best / np.linalg.norm(best, axis=1, keepdims=True)

    norms = np.linalg.norm(best, axis=1)
    valid = (norms > 1e-10) & np.all(np.isfinite(best), axis=1)
    if not np.all(valid):
        best = fibonacci_sphere(N)

    return best
# EVOLVE-BLOCK-END
