"""Rewritten program for covering the sphere with 50 equal caps."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull


def _covering_radius_degrees(pts):
    """Compute the covering radius in degrees for points on the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 90.0, -1, None
    
    facets = hull.simplices
    max_circumradius = 0.0
    worst_facet_idx = -1
    worst_cc = None
    
    for idx, facet in enumerate(facets):
        p1, p2, p3 = pts[facet[0]], pts[facet[1]], pts[facet[2]]
        
        a = p1
        b = p2
        c = p3
        
        ab = np.dot(a, b)
        bc = np.dot(b, c)
        ca = np.dot(c, a)
        
        coeff_a = 1.0 - bc * bc
        coeff_b = 1.0 - ca * ca
        coeff_c = 1.0 - ab * ab
        
        cc_vec = coeff_a * a + coeff_b * b + coeff_c * c
        cc_norm = np.linalg.norm(cc_vec)
        
        if cc_norm < 1e-10:
            continue
            
        cc = cc_vec / cc_norm
        
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            continue
        n = n / n_norm
        
        if np.dot(cc, n) < 0:
            cc = -cc
        
        cos_theta = np.clip(np.dot(cc, a), -1.0, 1.0)
        theta = np.arccos(cos_theta)
        
        if theta > max_circumradius:
            max_circumradius = theta
            worst_facet_idx = idx
            worst_cc = cc
    
    return np.degrees(max_circumradius), worst_facet_idx, worst_cc


def _force_dynamics(pts, steps=1000, k=6):
    """Run repulsive force dynamics."""
    n = len(pts)
    best_pts = pts.copy()
    best_radius, _, _ = _covering_radius_degrees(pts)
    
    for step in range(steps):
        dots = np.clip(pts @ pts.T, -1.0, 1.0)
        dists = np.arccos(dots)
        np.fill_diagonal(dists, np.inf)
        
        forces = np.zeros_like(pts)
        for i in range(n):
            nn_idx = np.argsort(dists[i])[:k]
            local_force = np.zeros(3)
            for j in nn_idx:
                if j == i: continue
                d = dists[i, j]
                if d < 1e-10: continue
                diff = pts[i] - pts[j]
                proj = diff - np.dot(diff, pts[i]) * pts[i]
                proj_norm = np.linalg.norm(proj)
                if proj_norm < 1e-10: continue
                force_mag = 1.0 / (d * d)
                local_force += force_mag * (proj / proj_norm)
            forces[i] = local_force
        
        current_dt = 0.01 * (1.0 - step / steps)
        if current_dt < 0.001: current_dt = 0.001
        
        new_pts = pts.copy()
        for i in range(n):
            f = forces[i]
            f_tangent = f - np.dot(f, pts[i]) * pts[i]
            new_pts[i] = pts[i] + current_dt * f_tangent
            norm = np.linalg.norm(new_pts[i])
            if norm > 1e-10:
                new_pts[i] = new_pts[i] / norm
            else:
                new_pts[i] = pts[i]
        
        pts = new_pts
        
        if step % 50 == 0:
            r, _, _ = _covering_radius_degrees(pts)
            if r < best_radius:
                best_radius = r
                best_pts = pts.copy()
                
    return best_pts


def _local_search(pts, max_iters=200):
    """Local search on the covering radius."""
    current_pts = pts.copy()
    current_radius, _, _ = _covering_radius_degrees(current_pts)
    
    for _ in range(max_iters):
        _, worst_idx, _ = _covering_radius_degrees(current_pts)
        if worst_idx == -1: break
        
        try:
            hull = ConvexHull(current_pts)
        except Exception:
            break
            
        facet = hull.simplices[worst_idx]
        v_idx = facet[np.random.randint(3)]
        
        noise = np.random.normal(0, 0.01, 3)
        new_point = current_pts[v_idx] + noise
        new_point = new_point / np.linalg.norm(new_point)
        
        test_pts = current_pts.copy()
        test_pts[v_idx] = new_point
        
        new_radius, _, _ = _covering_radius_degrees(test_pts)
        
        if new_radius < current_radius:
            current_pts = test_pts
            current_radius = new_radius
            
    return current_pts


def construct():
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    
    # Phase 1: Force Dynamics
    pts = _force_dynamics(pts, steps=1000, k=6)
    
    # Phase 2: Local Search
    pts = _local_search(pts, max_iters=200)
    
    # Final normalization
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    
    return pts
# EVOLVE-BLOCK-END
