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

    def smooth_repulsive_energy(params, beta=10.0):
        """
        Smooth energy combining pairwise repulsion and pressure from worst facets.
        Uses soft-max to smoothly weight the contribution of each facet circumcenter.
        """
        if time.time() > deadline - 10:
            return 1e10

        pts = params.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        pts = pts / norms

        # Pairwise repulsion: sum over all pairs of 1/sin^2(theta/2)
        # sin(theta/2) = ||p_i - p_j|| / 2
        pairwise_energy = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(pts[i] - pts[j])
                # Avoid division by zero; use a small epsilon
                sin_half = np.clip(d / 2.0, 1e-6, 1.0)
                pairwise_energy += 1.0 / (sin_half * sin_half)

        # Pressure from facet circumcenters: soft-max weighted sum
        facets = compute_facet_circumcenters(pts)
        if not facets:
            return pairwise_energy

        # Compute soft-max weights for facet radii
        radii = np.array([f[0] for f in facets])
        # Shift for numerical stability
        radii_shifted = radii - np.max(radii)
        weights = np.exp(beta * radii_shifted)
        weights /= np.sum(weights)

        # Weighted sum of distances from points to circumcenters
        pressure_energy = 0.0
        for w, (radius, center, simplex) in zip(weights, facets):
            for i in range(N):
                d = np.linalg.norm(pts[i] - center)
                sin_half = np.clip(d / 2.0, 1e-6, 1.0)
                pressure_energy += w * (1.0 / (sin_half * sin_half))

        return pairwise_energy + 0.1 * pressure_energy

    def smooth_repulsive_gradient(params, beta=10.0):
        """
        Numerical gradient of the smooth repulsive energy.
        """
        eps = 1e-6
        grad = np.zeros_like(params)
        f0 = smooth_repulsive_energy(params, beta)
        for k in range(len(params)):
            params[k] += eps
            f1 = smooth_repulsive_energy(params, beta)
            params[k] -= eps
            grad[k] = (f1 - f0) / eps

        return grad

    def optimize_repulsive_force(pts, lr=0.01, iterations=100):
        """
        Gradient descent on the smooth repulsive energy.
        """
        pts = pts.copy()
        params = pts.flatten()
        prev_energy = smooth_repulsive_energy(params)
        
        for it in range(iterations):
            if time.time() > deadline - 5:
                break

            grad = smooth_repulsive_gradient(params)
            params = params - lr * grad
            
            # Project back to sphere
            pts = params.reshape(N, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms = np.where(norms < 1e-10, 1.0, norms)
            pts = pts / norms
            params = pts.flatten()

            energy = smooth_repulsive_energy(params)
            if energy >= prev_energy:
                # Reduce learning rate
                lr *= 0.5
                if lr < 1e-10:
                    break
            else:
                prev_energy = energy

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

    # Phase 2: Smooth repulsive force optimization
    if time.time() < deadline - 30:
        try:
            # Multiple runs with different learning rates
            for lr in [0.05, 0.02, 0.01, 0.005]:
                if time.time() > deadline - 25:
                    break
                repulsive_pts = optimize_repulsive_force(current, lr=lr, iterations=80)
                repulsive_radius, _ = compute_covering_radius(repulsive_pts)
                if repulsive_radius < best_radius:
                    best_radius = repulsive_radius
                    best = repulsive_pts.copy()
                    current = repulsive_pts.copy()
        except Exception:
            pass

    # Phase 3: Worst-facet optimization with adaptive step size
    step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005]
    for step in step_sizes:
        if time.time() > deadline - 15:
            break

        # Move vertices of worst facet towards worst center
        current = current.copy()
        prev_radius = np.pi
        for _ in range(80):
            if time.time() > deadline - 15:
                break

            radius, worst_center = compute_covering_radius(current)

            # If radius hasn't improved, reduce step size
            if radius >= prev_radius:
                step *= 0.5
                if step < 1e-8:
                    break
            else:
                prev_radius = radius

            # Find the facet corresponding to worst_center
            try:
                hull = ConvexHull(current)
            except:
                break

            target_indices = None
            for simplex in hull.simplices:
                v1 = current[simplex[0]]
                v2 = current[simplex[1]]
                v3 = current[simplex[2]]
                edge1 = v2 - v1
                edge2 = v3 - v1
                n = np.cross(edge1, edge2)
                norm_n = np.linalg.norm(n)
                if norm_n < 1e-12:
                    continue
                c = n / norm_n
                if np.dot(c, v1) < 0:
                    c = -c
                if np.dot(c, worst_center) > 0.9999:
                    target_indices = simplex
                    break

            if target_indices is None:
                break

            # Move vertices towards worst_center in tangent space
            for idx in target_indices:
                v = current[idx]
                dir_vec = worst_center - v
                # Project onto tangent space: remove radial component
                radial_comp = np.dot(dir_vec, v)
                dir_vec = dir_vec - radial_comp * v
                norm_dir = np.linalg.norm(dir_vec)
                if norm_dir > 1e-10:
                    dir_vec /= norm_dir
                    current[idx] = v + step * dir_vec
                    current[idx] /= np.linalg.norm(current[idx])

        radius, _ = compute_covering_radius(current)

        if radius < best_radius:
            best_radius = radius
            best = current.copy()
        else:
            current = best.copy()

    # Phase 4: Random restarts with small perturbations
    if time.time() < deadline - 20:
        for restart in range(15):
            if time.time() > deadline - 20:
                break

            # Create a perturbed version of the best solution
            perturbed = best.copy()
            noise_scale = 0.03 / (restart + 1)
            for i in range(N):
                rand_vec = np.random.randn(3)
                rand_vec /= np.linalg.norm(rand_vec)
                perturbed[i] = perturbed[i] + noise_scale * rand_vec
                perturbed[i] /= np.linalg.norm(perturbed[i])

            # Optimize the perturbed solution
            for step in [0.02, 0.01]:
                if time.time() > deadline - 20:
                    break
                perturbed = perturbed.copy()
                prev_radius = np.pi
                for _ in range(40):
                    if time.time() > deadline - 20:
                        break
                    radius, worst_center = compute_covering_radius(perturbed)
                    if radius >= prev_radius:
                        step *= 0.5
                        if step < 1e-8:
                            break
                    else:
                        prev_radius = radius

                    try:
                        hull = ConvexHull(perturbed)
                    except:
                        break

                    target_indices = None
                    for simplex in hull.simplices:
                        v1 = perturbed[simplex[0]]
                        v2 = perturbed[simplex[1]]
                        v3 = perturbed[simplex[2]]
                        edge1 = v2 - v1
                        edge2 = v3 - v1
                        n = np.cross(edge1, edge2)
                        norm_n = np.linalg.norm(n)
                        if norm_n < 1e-12:
                            continue
                        c = n / norm_n
                        if np.dot(c, v1) < 0:
                            c = -c
                        if np.dot(c, worst_center) > 0.9999:
                            target_indices = simplex
                            break

                    if target_indices is None:
                        break

                    for idx in target_indices:
                        v = perturbed[idx]
                        dir_vec = worst_center - v
                        radial_comp = np.dot(dir_vec, v)
                        dir_vec = dir_vec - radial_comp * v
                        norm_dir = np.linalg.norm(dir_vec)
                        if norm_dir > 1e-10:
                            dir_vec /= norm_dir
                            perturbed[idx] = v + step * dir_vec
                            perturbed[idx] /= np.linalg.norm(perturbed[idx])

            radius, _ = compute_covering_radius(perturbed)
            if radius < best_radius:
                best_radius = radius
                best = perturbed.copy()

    # Phase 5: Final high-precision polish via adaptive step size in tangent space
    if time.time() < deadline - 5:
        final = best.copy()
        step = 0.005
        prev_radius = compute_covering_radius(final)[0]
        for _ in range(200):
            if time.time() > deadline - 5:
                break

            radius, worst_center = compute_covering_radius(final)

            if radius >= prev_radius:
                step *= 0.5
                if step < 1e-10:
                    break
            else:
                prev_radius = radius

            try:
                hull = ConvexHull(final)
            except:
                break

            target_indices = None
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
                if np.dot(c, worst_center) > 0.9999:
                    target_indices = simplex
                    break

            if target_indices is None:
                break

            for idx in target_indices:
                v = final[idx]
                dir_vec = worst_center - v
                radial_comp = np.dot(dir_vec, v)
                dir_vec = dir_vec - radial_comp * v
                dir_vec_norm = np.linalg.norm(dir_vec)
                if dir_vec_norm > 1e-10:
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
        best = fibonacci_sphere(N)

    return best
# EVOLVE-BLOCK-END
