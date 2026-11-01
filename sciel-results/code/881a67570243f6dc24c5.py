import numpy as np
from scipy.spatial import ConvexHull
import time

N = 50

def construct():
    # Initial Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_pts = pts.copy()
    best_radius = float('inf')
    
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    def compute_hull_data(p):
        """Compute hull facets and their circumcenters/radii."""
        try:
            hull = ConvexHull(p)
            facets = hull.simplices
        except Exception:
            return None, None, None
        
        circumcenters = np.zeros((len(facets), 3))
        radii = np.zeros(len(facets))
        
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
        # dE/dr_i = r_i * exp(beta * r_i^2) / sum(exp(beta * r_j^2))
        r2 = radii * radii
        max_r2 = np.max(r2)
        exp_terms = np.exp(beta * (r2 - max_r2))
        sum_exp = np.sum(exp_terms)
        weights = radii * exp_terms / sum_exp  # dE/dr_i
        
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
        # Pairwise distances
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
    
    # Phase 1: Coarse optimization
    eta = 0.02
    gamma = 1.0
    beta = 30.0
    
    for step in range(1500):
        if time.time() > deadline - 15.0:  # Reserve 15s for polish
            break
        
        forces, facets, circumcenters, radii = compute_forces(pts, beta)
        if forces is None:
            break
        
        # Update
        pts += eta * forces
        
        # Normalize
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        pts = pts / norms
        
        # Evaluate
        if radii is not None:
            current_radius = np.max(radii)
            if current_radius < best_radius:
                best_radius = current_radius
                best_pts = pts.copy()
        
        # Anneal
        eta *= 0.998
        beta += 0.5  # Increase sharpness of soft-max
    
    # Phase 2: Fine polish
    if time.time() < deadline - 3.0:
        pts = best_pts.copy()
        eta = 0.005
        beta = 100.0
        
        for step in range(500):
            if time.time() > deadline - 2.0:
                break
            
            forces, facets, circumcenters, radii = compute_forces(pts, beta)
            if forces is None:
                break
            
            pts += eta * forces
            
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms[norms < 1e-12] = 1.0
            pts = pts / norms
            
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
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    return best_pts
