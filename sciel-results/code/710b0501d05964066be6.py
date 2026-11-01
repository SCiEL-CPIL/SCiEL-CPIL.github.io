import time
import numpy as np
from scipy.spatial import ConvexHull

N = 130
_EPS = 1e-12

def _initial_fib():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _get_hull_data(pts):
    """
    Compute convex hull and extract facet normals, radii, and vertex indices.
    Returns:
        radii: (F,) array of circumradii in radians for each facet
        normals: (F, 3) array of outward unit normals
        faces: (F, 3) array of vertex indices for each facet
        valid_mask: boolean mask for valid facets
    """
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        
        AB = B - A
        AC = C - A
        cross = np.cross(AB, AC)
        area = 0.5 * np.linalg.norm(cross, axis=1)
        
        valid = area > _EPS
        if not np.any(valid):
            return np.array([]), np.array([]), np.array([]), np.array([], dtype=bool)

        normals = cross[valid]
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        
        centroids = (A[valid] + B[valid] + C[valid]) / 3.0
        signs = np.sum(normals * centroids, axis=1, keepdims=True)
        normals = normals * np.sign(signs)
        normals = normals / (norms + _EPS)
        
        cos_theta = np.sum(normals * A[valid], axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        radii = np.arccos(cos_theta)
        
        return radii, normals, faces[valid], valid
    except Exception:
        return np.array([]), np.array([]), np.array([]), np.array([], dtype=bool)

def _hull_radius(pts):
    """Compute the exact covering radius using the convex hull method."""
    radii, _, _, _ = _get_hull_data(pts)
    if len(radii) == 0:
        return np.pi
    return np.max(radii)

def _relax_worst_facet(pts, faces, normals, radii, worst_idx, step_size):
    """
    Attempt to reduce the radius of the worst facet by moving its vertices 
    towards the facet's normal (pole).
    """
    v_indices = faces[worst_idx]
    pole = normals[worst_idx]
    
    cand = pts.copy()
    # Move each vertex of the worst facet towards the pole
    # This tends to shrink the triangle and reduce its circumradius
    for v in v_indices:
        dir_vec = pole - pts[v]
        dir_norm = np.linalg.norm(dir_vec)
        if dir_norm > _EPS:
            dir_vec /= dir_norm
            cand[v] = pts[v] + step_size * dir_vec
            cand[v] /= (np.linalg.norm(cand[v]) + _EPS)
            
    return cand

def _optimize_local(pts, deadline):
    """
    Local optimization loop:
    1. Compute exact hull and find worst facet.
    2. Try to relax the worst facet by moving vertices towards its pole.
    3. If global radius improves, accept.
    4. If stuck, try random perturbations or smaller steps.
    """
    best_pts = pts.copy()
    best_r = _hull_radius(best_pts)
    
    # Adaptive step size
    step_size = 0.05
    stagnant_count = 0
    max_stagnant = 20
    
    while time.time() < deadline - 5.0:
        radii, normals, faces, valid = _get_hull_data(pts)
        if len(radii) == 0:
            break
            
        curr_r = np.max(radii)
        
        # If we found a better global radius, update best
        if curr_r < best_r - 1e-12:
            best_r = curr_r
            best_pts = pts.copy()
            stagnant_count = 0
        else:
            stagnant_count += 1
            if stagnant_count > max_stagnant:
                # Perturb to escape local minimum
                rng = np.random.default_rng()
                pts = pts + rng.standard_normal((N, 3)) * 0.01
                pts /= (np.linalg.norm(pts, axis=1, keepdims=True) + _EPS)
                stagnant_count = 0
                step_size = max(0.01, step_size * 0.8)
                continue
        
        # Find the worst facet
        worst_idx = np.argmax(radii)
        
        # Try to relax
        cand = _relax_worst_facet(pts, faces, normals, radii, worst_idx, step_size)
        
        # Evaluate candidate
        cand_r = _hull_radius(cand)
        
        if cand_r < curr_r - 1e-14:
            pts = cand
            # If improvement is small, reduce step size
            if curr_r - cand_r < 1e-5:
                step_size *= 0.9
        else:
            # If relaxation failed, try a smaller step or different direction
            step_size *= 0.7
            if step_size < 1e-4:
                # Stuck in a very tight local minimum, perturb
                rng = np.random.default_rng()
                # Perturb only the vertices of the worst facet
                for v in faces[worst_idx]:
                    pts[v] = pts[v] + rng.standard_normal(3) * 0.005
                    pts[v] /= (np.linalg.norm(pts[v]) + _EPS)
                step_size = 0.05
                stagnant_count += 1

    return best_pts, best_r

def construct():
    deadline = time.time() + 145.0
    
    # Start with Fibonacci spiral
    best_pts = _initial_fib()
    best_r = _hull_radius(best_pts)
    
    # Multiple restarts with random perturbations to find better basins
    n_restarts = 5
    time_per_restart = (deadline - 20.0) / n_restarts
    
    for restart in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        
        if restart == 0:
            pts = best_pts.copy()
        else:
            rng = np.random.default_rng(100 + restart * 7)
            # Larger perturbation for later restarts to explore more
            scale = 0.2 + 0.1 * restart
            pts = best_pts + rng.standard_normal((N, 3)) * scale
            pts /= (np.linalg.norm(pts, axis=1, keepdims=True) + _EPS)
            
        restart_deadline = min(deadline - 10.0, time.time() + time_per_restart)
        
        # Run local exact-geometry optimization
        pts, r = _optimize_local(pts, restart_deadline)
        
        if r < best_r - 1e-12:
            best_r = r
            best_pts = pts
    
    # Final high-precision polish on the best result found
    if time.time() < deadline - 5.0:
        best_pts, best_r = _optimize_local(best_pts, deadline - 2.0)
        
    # Final normalization and validity check
    best_pts = np.array(best_pts, dtype=float)
    best_pts /= (np.linalg.norm(best_pts, axis=1, keepdims=True) + _EPS)
    
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (N, 3):
        return _initial_fib()
        
    return best_pts
