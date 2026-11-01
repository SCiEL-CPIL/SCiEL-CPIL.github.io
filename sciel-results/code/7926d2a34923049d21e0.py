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

    def compute_facet_circumcenters(pts):
        try:
            hull = ConvexHull(pts)
        except Exception:
            return [(np.pi, np.array([0.0, 0.0, 1.0]), None)]

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

        if not results:
            return [(np.pi, np.array([0.0, 0.0, 1.0]), None)]

        return results

    def compute_covering_radius(pts):
        facets = compute_facet_circumcenters(pts)
        max_radius = 0.0
        worst_center = None
        for radius, c, _ in facets:
            if radius > max_radius:
                max_radius = radius
                worst_center = c
        if worst_center is None:
            return np.pi, np.array([0.0, 0.0, 1.0])
        return max_radius, worst_center

    def log_det_gram(pts):
        """
        Compute log of determinant of the Gram matrix (P @ P.T).
        This is a measure of uniformity: maximized when points are uniformly distributed.
        We minimize the negative of this to maximize uniformity.
        """
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        G = pts @ pts.T
        # Add small regularization for numerical stability
        G += np.eye(N) * 1e-10
        sign, logdet = np.linalg.slogdet(G)
        if sign <= 0:
            return 0.0
        return logdet

    def gram_based_objective(flat_pts):
        """
        Objective: minimize negative log-det of Gram matrix (maximize uniformity)
        plus a term that penalizes large covering radius.
        """
        pts = flat_pts.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        pts = pts / norms

        # Negative log-det (we want to maximize log-det, so minimize -log-det)
        neg_logdet = -log_det_gram(pts)

        # Covering radius penalty
        facets = compute_facet_circumcenters(pts)
        if not facets:
            covering_term = np.pi
        else:
            radii = np.array([r for r, _, _ in facets])
            max_r = np.max(radii)
            # Soft max for differentiability
            beta = 30.0
            if max_r > 0:
                soft_max = max_r + np.log(np.sum(np.exp((radii - max_r) * beta)) / len(radii)) / beta
            else:
                soft_max = 0.0
            covering_term = soft_max

        return neg_logdet * 0.1 + covering_term

    def gradient_gram(flat_pts):
        """
        Analytic gradient of the negative log-det of the Gram matrix.
        G = P P^T, d(-log det G)/dP = -2 G^{-1} P
        """
        pts = flat_pts.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        pts = pts / norms

        G = pts @ pts.T
        G += np.eye(N) * 1e-10
        try:
            G_inv = np.linalg.inv(G)
        except:
            return np.zeros_like(flat_pts)

        # Gradient of -log det G w.r.t. P is -2 * G_inv @ P
        grad = -2.0 * (G_inv @ pts)
        return grad.flatten()

    def optimize_gram_based(pts, max_iter=200, step_size=0.05):
        """
        Optimize using gradient descent on the Gram matrix objective.
        """
        pts = pts.copy()
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break

            flat = pts.flatten()
            val = gram_based_objective(flat)
            grad = gradient_gram(flat)

            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-12:
                break
            grad = grad / grad_norm

            # Adaptive step size
            new_pts = (flat - step_size * grad).reshape(N, 3)
            new_pts = new_pts / np.linalg.norm(new_pts, axis=1, keepdims=True)

            new_val = gram_based_objective(new_pts.flatten())
            if new_val < val:
                pts = new_pts
            else:
                step_size *= 0.5
                if step_size < 1e-6:
                    break

        return pts

    def simulated_annealing(pts, max_iter=5000, T_start=0.5, T_end=0.01):
        """
        Simulated annealing for global exploration.
        """
        pts = pts.copy()
        best_pts = pts.copy()
        best_radius = compute_covering_radius(pts)[0]
        T = T_start
        cooling = (T_end / T_start) ** (1.0 / max_iter)

        for it in range(max_iter):
            if time.time() > deadline - 10:
                break

            # Generate a perturbation
            idx = np.random.randint(N)
            angle = np.random.uniform(0.01, 0.3)
            # Random direction perpendicular to pts[idx]
            rand_vec = np.random.randn(3)
            rand_vec -= np.dot(rand_vec, pts[idx]) * pts[idx]
            rand_vec /= np.linalg.norm(rand_vec)

            new_pts = pts.copy()
            new_pts[idx] = pts[idx] * np.cos(angle) + rand_vec * np.sin(angle)
            new_pts[idx] /= np.linalg.norm(new_pts[idx])

            new_radius = compute_covering_radius(new_pts)[0]
            delta = new_radius - best_radius

            # Accept if better, or with probability exp(-delta/T)
            if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
                pts = new_pts
                if new_radius < best_radius:
                    best_radius = new_radius
                    best_pts = pts.copy()

            T *= cooling

        return best_pts

    def local_search_facet(pts, max_iter=100):
        """
        Local search: move vertices of worst facet toward its circumcenter.
        """
        pts = pts.copy()
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break

            radius, worst_center = compute_covering_radius(pts)

            try:
                hull = ConvexHull(pts)
            except:
                break

            target_indices = None
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
                if np.dot(c, worst_center) > 0.9999:
                    target_indices = simplex
                    break

            if target_indices is None:
                break

            # Move vertices toward circumcenter
            moved = False
            for idx in target_indices:
                v = pts[idx]
                dir_vec = worst_center - v
                norm_dir = np.linalg.norm(dir_vec)
                if norm_dir > 1e-10:
                    dir_vec /= norm_dir
                    pts[idx] = v + 0.02 * dir_vec
                    pts[idx] /= np.linalg.norm(pts[idx])
                    moved = True

            if not moved:
                break

        return pts

    # Phase 1: Generate multiple initial configurations
    best = None
    best_radius = np.pi

    initial_configs = []
    for offset in [0.5, 0.3, 0.7, 0.1, 0.9]:
        initial_configs.append(fibonacci_sphere(N, offset))

    # Add some random configurations
    for _ in range(3):
        rand_pts = np.random.randn(N, 3)
        rand_pts = rand_pts / np.linalg.norm(rand_pts, axis=1, keepdims=True)
        initial_configs.append(rand_pts)

    # Phase 2: For each initial config, run SA + Gram optimization + local search
    for config in initial_configs:
        if time.time() > deadline - 20:
            break

        current = config.copy()

        # Simulated annealing for global exploration
        if time.time() < deadline - 20:
            current = simulated_annealing(current, max_iter=3000, T_start=0.3, T_end=0.02)

        # Gram-based optimization for uniformity
        if time.time() < deadline - 15:
            current = optimize_gram_based(current, max_iter=100, step_size=0.03)

        # Local search on worst facet
        if time.time() < deadline - 10:
            current = local_search_facet(current, max_iter=80)

        radius, _ = compute_covering_radius(current)
        if radius < best_radius:
            best_radius = radius
            best = current.copy()

    # Phase 3: Refine the best solution with multiple local search passes
    if time.time() < deadline - 10:
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            best = local_search_facet(best, max_iter=50)
            radius, _ = compute_covering_radius(best)
            if radius < best_radius:
                best_radius = radius

    # Phase 4: Final fine-tuning
    if time.time() < deadline - 5:
        final = best.copy()
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            radius, worst_center = compute_covering_radius(final)

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
                norm_dir = np.linalg.norm(dir_vec)
                if norm_dir > 1e-10:
                    dir_vec /= norm_dir
                    final[idx] = v + 0.005 * dir_vec
                    final[idx] /= np.linalg.norm(final[idx])

        radius, _ = compute_covering_radius(final)
        if radius < best_radius:
            best = final

    # Final normalization and validation
    best = best / np.linalg.norm(best, axis=1, keepdims=True)

    norms = np.linalg.norm(best, axis=1)
    valid = (norms > 1e-10) & np.all(np.isfinite(best), axis=1)
    if not np.all(valid):
        best = fibonacci_sphere(N)

    return best
# EVOLVE-BLOCK-END
