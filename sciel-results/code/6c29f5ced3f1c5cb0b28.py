import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s limit

    def fibonacci_spiral():
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
        circumcenters = np.zeros((n_facets, 3))
        radii = np.zeros(n_facets)

        for k in range(n_facets):
            a = p[facets[k, 0]]
            b = p[facets[k, 1]]
            c = p[facets[k, 2]]
            v1 = b - a
            v2 = c - a
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
        if sum_exp < 1e-300:
            return None, None, None, None
        weights = radii * exp_terms / sum_exp

        forces = np.zeros_like(p)

        for k in range(n_facets):
            w = weights[k]
            if w < 1e-15:
                continue
            cc = circumcenters[k]
            for idx in facets[k]:
                forces[idx] += w * (cc - p[idx])

        # Repulsive forces
        diffs = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist_sq = np.sum(diffs * diffs, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        inv_dist_sq = 1.0 / dist_sq
        np.fill_diagonal(inv_dist_sq, 0.0)
        forces_rep = np.sum(diffs * inv_dist_sq[:, :, np.newaxis], axis=1)
        forces += 0.05 * forces_rep

        return forces, facets, circumcenters, radii

    def tangent_space_project(pts, delta):
        """Project delta onto tangent space of sphere at each point."""
        # For each point p_i, project delta_i to be orthogonal to p_i
        dot = np.sum(pts * delta, axis=1, keepdims=True)
        return delta - dot * pts

    def objective_sum_sq(params):
        """Objective: sum of squared facet radii."""
        X = params.reshape(N, 3)
        X = normalize(X)

        facets, circumcenters, radii = compute_hull_data(X)
        if facets is None:
            return 1e10

        return np.sum(radii**2)

    def objective_soft_max(params, beta=100.0):
        """Objective: soft-max of facet radii squared."""
        X = params.reshape(N, 3)
        X = normalize(X)

        facets, circumcenters, radii = compute_hull_data(X)
        if facets is None:
            return 1e10

        r2 = radii * radii
        max_r2 = np.max(r2)
        if max_r2 > 1.0:
            return 1e10
        exp_terms = np.exp(beta * (r2 - max_r2))
        return (1.0 / beta) * np.log(np.sum(exp_terms)) + max_r2 / beta

    best_pts = fibonacci_spiral()
    best_radius = float('inf')

    # Evaluate initial
    facets, circumcenters, radii = compute_hull_data(best_pts)
    if radii is not None:
        best_radius = np.max(radii)

    # Phase 1: Multiple restarts with gradient descent
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() > deadline - 30.0:
            break

        if restart == 0:
            pts = fibonacci_spiral()
        else:
            # Add random perturbation to previous best
            pts = best_pts.copy()
            noise = np.random.randn(N, 3) * 0.1
            pts = normalize(pts + noise)

        eta = 0.02
        beta = 20.0

        for step in range(800):
            if time.time() > deadline - 25.0:
                break

            forces, facets, circumcenters, radii = compute_forces(pts, beta)
            if forces is None:
                break

            # Project to tangent space
            forces_tan = tangent_space_project(pts, forces)

            pts = normalize(pts + eta * forces_tan)

            if radii is not None:
                current_radius = np.max(radii)
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_pts = pts.copy()

            # Anneal
            eta *= 0.997
            beta += 0.3

    # Phase 2: L-BFGS-B on sum of squares
    if time.time() < deadline - 20.0:
        pts = best_pts.copy()

        try:
            res = minimize(objective_sum_sq, pts.flatten(), method='L-BFGS-B',
                           options={'maxiter': 30, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 100})

            if res.fun < objective_sum_sq(pts.flatten()):
                new_pts = normalize(res.x.reshape(N, 3))
                facets, circumcenters, radii = compute_hull_data(new_pts)
                if radii is not None:
                    current_radius = np.max(radii)
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_pts = new_pts.copy()
        except Exception:
            pass

    # Phase 3: L-BFGS-B on soft-max
    if time.time() < deadline - 10.0:
        pts = best_pts.copy()

        try:
            res = minimize(lambda x: objective_soft_max(x, beta=200.0), pts.flatten(),
                           method='L-BFGS-B',
                           options={'maxiter': 20, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 100})

            if res.fun < objective_soft_max(pts.flatten(), beta=200.0):
                new_pts = normalize(res.x.reshape(N, 3))
                facets, circumcenters, radii = compute_hull_data(new_pts)
                if radii is not None:
                    current_radius = np.max(radii)
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_pts = new_pts.copy()
        except Exception:
            pass

    # Phase 4: Fine polish with gradient descent on tangent space
    if time.time() < deadline - 3.0:
        pts = best_pts.copy()
        eta = 0.002
        beta = 300.0

        for step in range(400):
            if time.time() > deadline - 2.0:
                break

            forces, facets, circumcenters, radii = compute_forces(pts, beta)
            if forces is None:
                break

            forces_tan = tangent_space_project(pts, forces)
            pts = normalize(pts + eta * forces_tan)

            if radii is not None:
                current_radius = np.max(radii)
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_pts = pts.copy()

            eta *= 0.998

    # Final validation
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-10) or not np.all(np.isfinite(best_pts)):
        best_pts = fibonacci_spiral()

    return normalize(best_pts)
