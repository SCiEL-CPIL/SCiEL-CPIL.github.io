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

    def compute_facet_data(pts):
        """
        Compute all facet circumcenters, radii, and indices.
        Returns: list of (radius_radians, circumcenter, simplex_indices)
        """
        try:
            hull = ConvexHull(pts)
        except Exception:
            return []

        facets = []
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
            facets.append((radius, c, simplex))

        return facets

    def compute_covering_radius(pts):
        """
        Compute the covering radius using the convex hull facet circumradius method.
        Returns the covering radius in radians and the worst-case point (circumcenter of worst facet).
        """
        facets = compute_facet_data(pts)
        if not facets:
            return np.pi, np.array([0.0, 0.0, 1.0])

        max_radius = max(f[0] for f in facets)
        worst_facet = max(facets, key=lambda x: x[0])
        return max_radius, worst_facet[1]

    def smooth_objective(flat_pts):
        """
        Smooth surrogate: sum of squared facet circumradii.
        This is differentiable and amenable to L-BFGS-B.
        """
        pts = flat_pts.reshape(N, 3)
        # Normalize
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        pts = pts / norms

        facets = compute_facet_data(pts)
        if not facets:
            return 1e6  # Large penalty for invalid hull

        radii = np.array([f[0] for f in facets])
        return np.sum(radii ** 2)

    def smooth_gradient(flat_pts):
        """
        Numerical gradient of the smooth objective.
        """
        eps = 1e-7
        grad = np.zeros_like(flat_pts)
        f0 = smooth_objective(flat_pts)

        for i in range(len(flat_pts)):
            fp = flat_pts.copy()
            fm = flat_pts.copy()
            fp[i] += eps
            fm[i] -= eps
            grad[i] = (smooth_objective(fp) - smooth_objective(fm)) / (2 * eps)

        return grad

    def polish_max_radius(pts, max_iters=200, step_size=0.01):
        """
        Local polish: minimize the maximum facet radius using projected gradient steps.
        """
        pts = pts.copy()
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break

            facets = compute_facet_data(pts)
            if not facets:
                break

            max_radius = max(f[0] for f in facets)
            worst_facet = max(facets, key=lambda x: x[0])
            worst_center = worst_facet[1]
            target_indices = worst_facet[2]

            # Move vertices toward worst_center
            for idx in target_indices:
                v = pts[idx]
                dir_vec = worst_center - v
                norm_dir = np.linalg.norm(dir_vec)
                if norm_dir > 1e-10:
                    dir_vec /= norm_dir
                    # Project onto tangent space: remove radial component
                    radial = np.dot(dir_vec, v) * v
                    tangent_dir = dir_vec - radial
                    tangent_norm = np.linalg.norm(tangent_dir)
                    if tangent_norm > 1e-12:
                        tangent_dir /= tangent_norm
                        pts[idx] = v + step_size * tangent_dir
                        pts[idx] /= np.linalg.norm(pts[idx])

        return pts

    # Phase 1: Initial Fibonacci spiral
    current = fibonacci_sphere(N)
    best = current.copy()
    best_radius, _ = compute_covering_radius(best)

    # Phase 2: L-BFGS-B optimization on smooth surrogate
    if time.time() < deadline - 60:
        flat_init = current.flatten()
        try:
            result = minimize(
                smooth_objective,
                flat_init,
                method='L-BFGS-B',
                jac=smooth_gradient,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 5000}
            )
            if result.success or result.fun < smooth_objective(flat_init):
                candidate = result.x.reshape(N, 3)
                norms = np.linalg.norm(candidate, axis=1)
                valid = (norms > 1e-10) & np.all(np.isfinite(candidate), axis=1)
                if np.all(valid):
                    candidate = candidate / norms[:, np.newaxis]
                    rad, _ = compute_covering_radius(candidate)
                    if rad < best_radius:
                        best_radius = rad
                        best = candidate.copy()
                        current = candidate.copy()
        except Exception:
            pass

    # Phase 3: Local polish on max radius
    if time.time() < deadline - 20:
        current = best.copy()
        polished = polish_max_radius(current, max_iters=300, step_size=0.02)
        rad, _ = compute_covering_radius(polished)
        if rad < best_radius:
            best_radius = rad
            best = polished.copy()
            current = polished.copy()

    # Phase 4: Fine polish with smaller steps
    if time.time() < deadline - 10:
        current = best.copy()
        polished = polish_max_radius(current, max_iters=200, step_size=0.005)
        rad, _ = compute_covering_radius(polished)
        if rad < best_radius:
            best_radius = rad
            best = polished.copy()

    # Phase 5: Random restarts with small perturbations and polish
    if time.time() < deadline - 15:
        np.random.seed(42)
        for restart in range(10):
            if time.time() > deadline - 15:
                break

            perturbed = best.copy()
            noise_scale = 0.03 / (restart + 1)
            for i in range(N):
                rand_vec = np.random.randn(3)
                rand_vec /= np.linalg.norm(rand_vec)
                perturbed[i] = perturbed[i] + noise_scale * rand_vec
                perturbed[i] /= np.linalg.norm(perturbed[i])

            # Quick L-BFGS-B from perturbed start
            if time.time() < deadline - 25:
                flat_pert = perturbed.flatten()
                try:
                    result = minimize(
                        smooth_objective,
                        flat_pert,
                        method='L-BFGS-B',
                        jac=smooth_gradient,
                        options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 2000}
                    )
                    if result.fun < smooth_objective(flat_pert):
                        candidate = result.x.reshape(N, 3)
                        norms = np.linalg.norm(candidate, axis=1)
                        valid = (norms > 1e-10) & np.all(np.isfinite(candidate), axis=1)
                        if np.all(valid):
                            candidate = candidate / norms[:, np.newaxis]
                            perturbed = candidate.copy()
                except Exception:
                    pass

            # Polish
            if time.time() < deadline - 15:
                polished = polish_max_radius(perturbed, max_iters=100, step_size=0.01)
                rad, _ = compute_covering_radius(polished)
                if rad < best_radius:
                    best_radius = rad
                    best = polished.copy()

    # Final normalization and validation
    best = best / np.linalg.norm(best, axis=1, keepdims=True)

    # Ensure all points are non-zero and finite
    norms = np.linalg.norm(best, axis=1)
    valid = (norms > 1e-10) & np.all(np.isfinite(best), axis=1)
    if not np.all(valid):
        best = fibonacci_sphere(N)

    return best
# EVOLVE-BLOCK-END
