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

    def compute_hull_data(p):
        """Compute hull facets and their circumcenters/radii (vectorized)."""
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
        
        valid = norm_n > 1e-12
        circumcenters = np.zeros((n_facets, 3))
        radii = np.zeros(n_facets)
        
        if np.any(valid):
            cc = n[valid] / norm_n[valid, np.newaxis]
            dots = np.sum(cc * a[valid], axis=1)
            cc[dots < 0] *= -1
            circumcenters[valid] = cc
            
            cos_r = np.clip(np.sum(cc * a[valid], axis=1), -1.0, 1.0)
            radii[valid] = np.arccos(cos_r)
        
        return facets, circumcenters, radii

    def get_max_radius(pts):
        facets, circumcenters, radii = compute_hull_data(pts)
        if radii is None:
            return float('inf'), np.array([0.0, 0.0, 1.0])
        idx = np.argmax(radii)
        return radii[idx], circumcenters[idx]

    def soft_max_objective(params, beta=200.0):
        """
        Soft-max surrogate: approximates max(radii) via log-sum-exp.
        """
        X = params.reshape(N, 3)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        X = X / norms
        
        facets, circumcenters, radii = compute_hull_data(X)
        if facets is None:
            return 1e10
        
        r_max = np.max(radii)
        if r_max < 1e-15:
            return r_max
        shifted = beta * (radii - r_max)
        log_sum_exp = np.log(np.sum(np.exp(shifted))) / beta + r_max
        return log_sum_exp

    def repulsion_relax(pts, n_iters=800, alpha_start=0.08):
        """
        Repulsion relaxation with adaptive step size.
        Minimize sum 1/||p_i - p_j||^2.
        """
        pts = pts.copy()
        alpha = alpha_start
        
        def repulsion_energy(p):
            diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dist_sq = np.sum(diff * diff, axis=2)
            mask = ~np.eye(N, dtype=bool)
            inv_dist_sq = np.zeros_like(dist_sq)
            np.divide(1.0, dist_sq, out=inv_dist_sq, where=mask)
            return 0.5 * np.sum(inv_dist_sq)
        
        current_energy = repulsion_energy(pts)
        
        for it in range(n_iters):
            if time.time() > deadline - 5.0:
                break
                
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
            for _ in range(10):
                new_pts = pts + step * forces_tan
                norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
                norms[norms < 1e-12] = 1.0
                new_pts = new_pts / norms
                new_energy = repulsion_energy(new_pts)
                if new_energy < current_energy:
                    pts = new_pts
                    current_energy = new_energy
                    improved = True
                    break
                step *= 0.5
            
            if not improved:
                alpha *= 0.8
            else:
                alpha *= 1.01
            
            alpha = min(alpha, 0.1)
            alpha = max(alpha, 1e-6)
            
        return pts

    def lloyd_iteration(pts, n_iters=50):
        """
        Improved Lloyd's algorithm.
        """
        pts = pts.copy()
        
        for it in range(n_iters):
            if time.time() > deadline - 5.0:
                break
                
            facets, circumcenters, radii = compute_hull_data(pts)
            if facets is None:
                break
            
            max_rad = np.max(radii)
            if max_rad < 1e-10:
                break
            
            updates = np.zeros_like(pts)
            counts = np.zeros(N)
            
            for f in range(len(facets)):
                facet_pts = facets[f]
                cc = circumcenters[f]
                rad = radii[f]
                if rad < 1e-12:
                    continue
                weight = (rad / max_rad) ** 2
                for idx in facet_pts:
                    updates[idx] += weight * (cc - pts[idx])
                    counts[idx] += weight
            
            mask = counts > 1e-10
            if np.any(mask):
                step_size = 0.05 * (max_rad / (np.mean(radii) + 1e-15))
                step_size = min(step_size, 0.1)
                pts[mask] += step_size * updates[mask] / counts[mask, np.newaxis]
            
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms[norms < 1e-12] = 1.0
            pts = pts / norms
            
        return pts

    def optimize_worst_facet(pts, step_size, iterations=50):
        pts = pts.copy()
        for _ in range(iterations):
            if time.time() > deadline - 5:
                break

            radius, worst_center = get_max_radius(pts)

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

            for idx in target_indices:
                v = pts[idx]
                dir_vec = worst_center - v
                norm_dir = np.linalg.norm(dir_vec)
                if norm_dir > 1e-10:
                    dir_vec /= norm_dir
                    pts[idx] = v + step_size * dir_vec
                    pts[idx] /= np.linalg.norm(pts[idx])

        return pts

    # Phase 1: Initial Fibonacci spiral
    best = fibonacci_sphere(N, 0.5)
    best_radius, _ = get_max_radius(best)

    # Phase 2: Pre-processing with Lloyd and repulsion
    current = best.copy()
    if time.time() < deadline - 30:
        current = lloyd_iteration(current, iterations=30)
        radius, _ = get_max_radius(current)
        if radius < best_radius:
            best_radius = radius
            best = current.copy()

    if time.time() < deadline - 30:
        current = repulsion_relax(best.copy(), iterations=100)
        radius, _ = get_max_radius(current)
        if radius < best_radius:
            best_radius = radius
            best = current.copy()

    # Phase 3: L-BFGS-B optimization using soft-max surrogate
    if time.time() < deadline - 25:
        try:
            result = minimize(
                soft_max_objective,
                best.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            if result.success and result.x is not None:
                current = result.x.reshape(N, 3)
                current = current / np.linalg.norm(current, axis=1, keepdims=True)
                radius, _ = get_max_radius(current)
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
            radius, _ = get_max_radius(current)
            if radius < best_radius:
                best_radius = radius
                best = current.copy()

    # Phase 5: Random restarts with perturbations
    if time.time() < deadline - 20:
        for restart in range(10):
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
                try:
                    result = minimize(
                        soft_max_objective,
                        perturbed.flatten(),
                        method='L-BFGS-B',
                        options={'maxiter': 80, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
                    )
                    if result.success and result.x is not None:
                        perturbed = result.x.reshape(N, 3)
                        perturbed = perturbed / np.linalg.norm(perturbed, axis=1, keepdims=True)
                except:
                    pass

            if time.time() < deadline - 20:
                perturbed = optimize_worst_facet(perturbed, 0.01, iterations=30)

            radius, _ = get_max_radius(perturbed)
            if radius < best_radius:
                best_radius = radius
                best = perturbed.copy()

    # Phase 6: Final fine-tuning
    if time.time() < deadline - 5:
        final = best.copy()
        # High-precision polish with soft-max
        if time.time() < deadline - 3:
            try:
                result = minimize(
                    soft_max_objective,
                    final.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100}
                )
                if result.success and result.x is not None:
                    final = result.x.reshape(N, 3)
                    final = final / np.linalg.norm(final, axis=1, keepdims=True)
            except:
                pass
        
        # Final exact local search
        final = optimize_worst_facet(final, 0.001, iterations=50)
        
        radius, _ = get_max_radius(final)
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
