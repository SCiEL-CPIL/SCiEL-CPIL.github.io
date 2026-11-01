import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin

    def fibonacci_initial():
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
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

    def compute_forces(p, beta=50.0):
        """Compute smooth forces from soft-max energy + repulsion."""
        facets, circumcenters, radii = compute_hull_data(p)
        if facets is None:
            return None, None, None, None
        
        n_facets = len(facets)
        if n_facets == 0:
            return None, None, None, None
        
        # Soft-max energy: E = (1/beta) * log(sum(exp(beta * r_i^2)))
        # dE/dr_i = (2/beta) * r_i * exp(beta * r_i^2) / sum(exp(beta * r_j^2))
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
        diffs = p[:, np.newaxis, :] - p[np.newaxis, :, :]  # (N, N, 3)
        dist_sq = np.sum(diffs * diffs, axis=2)  # (N, N)
        np.fill_diagonal(dist_sq, 1.0)  # avoid division by zero
        
        # Repulsive force: F_i = sum_{j != i} (p_i - p_j) / d_ij^2
        # Vectorized: for each i, sum over j
        inv_dist_sq = 1.0 / dist_sq
        np.fill_diagonal(inv_dist_sq, 0.0)
        
        # forces_rep[i] = sum_j diffs[i,j] * inv_dist_sq[i,j]
        forces_rep = np.sum(diffs * inv_dist_sq[:, :, np.newaxis], axis=1)
        forces += 0.1 * forces_rep  # small repulsion to prevent coincidence
        
        return forces, facets, circumcenters, radii

    best_pts = fibonacci_initial()
    best_radius = float('inf')

    # Phase 1: Coarse optimization
    pts = best_pts.copy()
    eta = 0.02
    beta = 30.0
    
    for step in range(1500):
        if time.time() > deadline - 15.0:  # Reserve 15s for polish
            break
        
        forces, facets, circumcenters, radii = compute_forces(pts, beta)
        if forces is None:
            break
        
        # Update
        pts += eta * forces
        pts = normalize(pts)
        
        # Evaluate
        if radii is not None:
            current_radius = np.max(radii)
            if current_radius < best_radius:
                best_radius = current_radius
                best_pts = pts.copy()
        
        # Anneal
        eta *= 0.998
        beta += 0.5  # Increase sharpness of soft-max
    
    # Phase 2: L-BFGS-B Polish on Sum of Squared Radii
    # This helps establish a robust local structure by minimizing the sum of squares,
    # which is smoother than the max function.
    if time.time() < deadline - 10.0:
        pts = best_pts.copy()
        
        def objective_sum_sq(params):
            X = params.reshape(N, 3)
            X = normalize(X)
            
            facets, circumcenters, radii = compute_hull_data(X)
            if facets is None:
                return 1e10
            
            # Sum of squared radii + small variance penalty to encourage uniformity
            r = radii
            return np.sum(r**2) + 0.01 * np.var(r)
        
        # Use L-BFGS-B with more iterations for better convergence
        try:
            res = minimize(objective_sum_sq, pts.flatten(), method='L-BFGS-B', 
                           options={'maxiter': 80, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100})
            
            if res.success or res.fun < objective_sum_sq(pts.flatten()):
                new_pts = normalize(res.x.reshape(N, 3))
                
                # Evaluate new best
                facets, circumcenters, radii = compute_hull_data(new_pts)
                if radii is not None:
                    current_radius = np.max(radii)
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_pts = new_pts.copy()
                        pts = new_pts.copy()
        except Exception:
            pass

    # Phase 3: Projected Gradient Descent directly on the covering radius (minimax)
    # This phase directly minimizes the max angular distance by moving points
    # toward the circumcenters of the largest facets.
    if time.time() < deadline - 2.0:
        pts = best_pts.copy()
        eta = 0.003
        
        for step in range(800):
            if time.time() > deadline - 1.0:
                break
            
            facets, circumcenters, radii = compute_hull_data(pts)
            if facets is None:
                break
            
            current_radius = np.max(radii)
            if current_radius < best_radius:
                best_radius = current_radius
                best_pts = pts.copy()
            
            # Find the largest facet(s) and move their vertices toward the circumcenter
            # Use a soft threshold: move points in facets with radius > 0.95 * max
            threshold = 0.95 * current_radius
            active_facets = np.where(radii >= threshold)[0]
            
            if len(active_facets) == 0:
                break
            
            forces = np.zeros_like(pts)
            
            # For each active facet, compute the direction to move each vertex
            for k in active_facets:
                cc = circumcenters[k]
                facet_pts = pts[facets[k]]
                
                # Move each vertex toward the circumcenter, weighted by how far it is
                # from the circumcenter (in the tangent plane)
                for idx in facets[k]:
                    # Tangential component: project (cc - p[idx]) onto tangent plane at p[idx]
                    radial = np.dot(cc - pts[idx], pts[idx])
                    tangential = (cc - pts[idx]) - radial * pts[idx]
                    forces[idx] += tangential
            
            # Normalize forces to have unit norm per point
            for i in range(N):
                f_norm = np.linalg.norm(forces[i])
                if f_norm > 1e-12:
                    forces[i] /= f_norm
            
            # Apply update
            pts += eta * forces
            pts = normalize(pts)
            
            # Anneal step size
            eta *= 0.997
    
    # Phase 4: Final fine polish with soft-max gradient
    if time.time() < deadline - 1.0:
        pts = best_pts.copy()
        eta = 0.002
        beta = 150.0
        
        for step in range(300):
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
    
    # Final validation: ensure points are valid
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-10) or not np.all(np.isfinite(best_pts)):
        # Fallback to Fibonacci
        best_pts = fibonacci_initial()
    
    # Ensure exact normalization
    best_pts = normalize(best_pts)
    
    return best_pts
