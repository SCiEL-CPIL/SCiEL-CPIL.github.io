import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

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
        """Compute hull facets and their circumcenters/radii."""
        try:
            hull = ConvexHull(p)
            facets = hull.simplices
        except Exception:
            return None, None, None

        n_facets = len(facets)
        if n_facets == 0:
            return None, None, None

        circumcenters = np.zeros((n_facets, 3))
        radii = np.zeros(n_facets)

        for k, facet in enumerate(facets):
            a = p[facet[0]]
            b = p[facet[1]]
            c = p[facet[2]]
            v1 = a - b
            v2 = b - c
            n = np.cross(v1, v2)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-12:
                circumcenters[k] = a
                radii[k] = 0.0
                continue
            cc = n / norm_n
            if np.dot(cc, a) < 0:
                cc = -cc
            circumcenters[k] = cc
            cos_r = np.clip(np.dot(cc, a), -1.0, 1.0)
            radii[k] = np.arccos(cos_r)

        return facets, circumcenters, radii

    def compute_covering_radius(p):
        """Compute the covering radius from hull data."""
        facets, circumcenters, radii = compute_hull_data(p)
        if facets is None:
            return float('inf')
        return np.max(radii)

    def compute_forces(p, beta=50.0):
        """Compute smooth forces from soft-max energy + repulsion."""
        facets, circumcenters, radii = compute_hull_data(p)
        if facets is None:
            return None, None, None, None

        n_facets = len(facets)
        if n_facets == 0:
            return None, None, None, None

        # Soft-max energy: E = (1/beta) * log(sum(exp(beta * r_i^2)))
        r2 = radii * radii
        max_r2 = np.max(r2)
        exp_terms = np.exp(beta * (r2 - max_r2))
        sum_exp = np.sum(exp_terms)
        # Chain rule: dE/dr_i = (2/beta) * r_i * softmax_weight_i
        weights = (2.0 / beta) * radii * exp_terms / sum_exp

        forces = np.zeros_like(p)

        # Attractive forces from facet circumcenters (weighted by soft-max)
        for k in range(n_facets):
            w = weights[k]
            if w < 1e-15:
                continue
            cc = circumcenters[k]
            for idx in facets[k]:
                forces[idx] += w * (cc - p[idx])

        # Repulsive forces (vectorized)
        diffs = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist_sq = np.sum(diffs * diffs, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        inv_dist_sq = 1.0 / dist_sq
        np.fill_diagonal(inv_dist_sq, 0.0)
        forces_rep = np.sum(diffs * inv_dist_sq[:, :, np.newaxis], axis=1)
        forces += 0.05 * forces_rep

        return forces, facets, circumcenters, radii

    def optimize_from_initial(pts_init, time_budget):
        """Run multi-phase optimization from a given initial configuration."""
        best_pts = pts_init.copy()
        best_radius = compute_covering_radius(best_pts)
        end_time = start_time + time_budget

        # Phase 1: Coarse soft-max gradient descent
        pts = best_pts.copy()
        eta = 0.02
        beta = 25.0
        n_steps = 800

        for step in range(n_steps):
            if time.time() > end_time - 20.0:
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

            eta *= 0.997
            beta += 0.3

        # Phase 2: L-BFGS-B on sum of squared radii
        if time.time() < end_time - 15.0:
            pts = best_pts.copy()

            def objective_sum_sq(params):
                X = params.reshape(N, 3)
                X = normalize(X)
                facets, circumcenters, radii = compute_hull_data(X)
                if facets is None:
                    return 1e10
                return np.sum(radii**2) + 0.005 * np.var(radii)

            try:
                res = minimize(objective_sum_sq, pts.flatten(), method='L-BFGS-B',
                               options={'maxiter': 60, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 80})
                if res.success or res.fun < objective_sum_sq(pts.flatten()):
                    new_pts = normalize(res.x.reshape(N, 3))
                    current_radius = compute_covering_radius(new_pts)
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_pts = new_pts.copy()
                        pts = new_pts.copy()
            except Exception:
                pass

        # Phase 3: Direct minimax projected gradient
        if time.time() < end_time - 8.0:
            pts = best_pts.copy()
            eta = 0.003

            for step in range(600):
                if time.time() > end_time - 5.0:
                    break

                facets, circumcenters, radii = compute_hull_data(pts)
                if facets is None:
                    break

                current_radius = np.max(radii)
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_pts = pts.copy()

                threshold = 0.95 * current_radius
                active_facets = np.where(radii >= threshold)[0]

                if len(active_facets) == 0:
                    break

                forces = np.zeros_like(pts)

                for k in active_facets:
                    cc = circumcenters[k]
                    for idx in facets[k]:
                        radial = np.dot(cc - pts[idx], pts[idx])
                        tangential = (cc - pts[idx]) - radial * pts[idx]
                        forces[idx] += tangential

                for i in range(N):
                    f_norm = np.linalg.norm(forces[i])
                    if f_norm > 1e-12:
                        forces[i] /= f_norm

                pts += eta * forces
                pts = normalize(pts)
                eta *= 0.997

        # Phase 4: Fine soft-max polish
        if time.time() < end_time - 2.0:
            pts = best_pts.copy()
            eta = 0.002
            beta = 120.0

            for step in range(200):
                if time.time() > end_time - 1.0:
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

        return best_pts, best_radius

    # Multi-restart strategy
    # Restart 1: Fibonacci
    fib_pts = fibonacci_initial()
    best_pts, best_radius = optimize_from_initial(fib_pts, 95.0)

    # Restart 2: Random perturbation of best
    if time.time() < deadline - 30.0:
        rng = np.random.default_rng(42)
        perturbed = best_pts + rng.normal(0, 0.05, (N, 3))
        perturbed = normalize(perturbed)
        pts2, rad2 = optimize_from_initial(perturbed, 35.0)
        if rad2 < best_radius:
            best_radius = rad2
            best_pts = pts2

    # Restart 3: Different random initial
    if time.time() < deadline - 20.0:
        rng = np.random.default_rng(123)
        random_pts = rng.normal(0, 1, (N, 3))
        random_pts = normalize(random_pts)
        pts3, rad3 = optimize_from_initial(random_pts, 25.0)
        if rad3 < best_radius:
            best_radius = rad3
            best_pts = pts3

    # Final validation
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-10) or not np.all(np.isfinite(best_pts)):
        best_pts = fibonacci_initial()
        best_radius = compute_covering_radius(best_pts)

    best_pts = normalize(best_pts)
    return best_pts
