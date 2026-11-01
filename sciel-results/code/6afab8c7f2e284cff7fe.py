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

def _compute_hull_info(pts):
    """
    Compute convex hull and return information about facets.
    Returns:
        radii: array of angular circumradii for each facet
        faces: array of facet vertex indices
        normals: array of outward normals for each facet
        valid: boolean mask for non-degenerate facets
        max_idx: index of the facet with maximum radius
    """
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        M = len(faces)
        
        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        
        AB = B - A
        AC = C - A
        cross = np.cross(AB, AC)
        area = 0.5 * np.linalg.norm(cross, axis=1)
        
        valid = area > _EPS
        if not np.any(valid):
            return None, None, None, None, None
            
        normals = cross[valid]
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        centroids = (A[valid] + B[valid] + C[valid]) / 3.0
        signs = np.sum(normals * centroids, axis=1, keepdims=True)
        normals = normals * np.sign(signs)
        normals = normals / (norms + _EPS)
        
        cos_theta = np.sum(normals * A[valid], axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        radii = np.arccos(cos_theta)
        
        max_idx = np.argmax(radii)
        
        return radii, faces[valid], normals, valid, max_idx
    except Exception:
        return None, None, None, None, None

def _move_towards_pole(pts, vertex_idx, pole, step_size):
    """
    Move a point on the sphere towards a target pole along the geodesic.
    """
    v = pts[vertex_idx]
    # Vector from v to pole in tangent space approximation or direct spherical interpolation
    # Using simple vector addition and normalization is a common heuristic for small steps
    # For larger steps, slerp is better, but for local polish, small steps make them similar.
    # We use: new_v = normalize(v + step * (pole - v))
    # This is equivalent to moving along the chord, which projects to a geodesic-like path.
    dir_vec = pole - v
    dir_norm = np.linalg.norm(dir_vec)
    if dir_norm > _EPS:
        # Normalize direction to ensure step_size is in "radians" roughly?
        # Actually, v + step*(pole-v) moves a distance of step*||pole-v|| in Euclidean space.
        # For small angles, Euclidean distance ~ angular distance.
        # To be precise about angular step, we can use slerp.
        # But let's stick to the simple projection first as it's robust.
        new_v = v + step_size * dir_vec
        new_v /= (np.linalg.norm(new_v) + _EPS)
    else:
        new_v = v
    return new_v

def _local_polish(pts, deadline):
    """
    Perform local optimization by iteratively reducing the worst facet's radius.
    """
    best_pts = pts.copy()
    radii, faces, normals, valid, max_idx = _compute_hull_info(pts)
    if radii is None:
        return pts
    best_r = np.max(radii)
    
    step_size = 0.01
    no_improve_count = 0
    max_no_improve = 20
    
    while time.time() < deadline:
        radii, faces, normals, valid, max_idx = _compute_hull_info(pts)
        if radii is None:
            break
            
        curr_r = np.max(radii)
        
        if curr_r < best_r:
            best_r = curr_r
            best_pts = pts.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                step_size *= 0.9
                no_improve_count = 0
                if step_size < 1e-5:
                    break
        
        # Identify the worst facet
        worst_idx = max_idx
        worst_face = faces[worst_idx]
        worst_pole = normals[worst_idx]
        
        # Attempt to reduce the radius of this facet by moving its vertices towards its pole
        # We try moving one vertex at a time to see which gives the best improvement
        best_candidate = pts
        best_candidate_r = curr_r
        
        for v_idx in worst_face:
            cand = pts.copy()
            cand[v_idx] = _move_towards_pole(pts, v_idx, worst_pole, step_size)
            
            cand_radii, _, _, _, _ = _compute_hull_info(cand)
            if cand_radii is not None:
                cand_r = np.max(cand_radii)
                if cand_r < best_candidate_r:
                    best_candidate_r = cand_r
                    best_candidate = cand
        
        if best_candidate_r < curr_r:
            pts = best_candidate
        else:
            # If no single vertex move helped, try a smaller step or break
            step_size *= 0.8
            if step_size < 1e-5:
                break

    return best_pts

def construct():
    deadline = time.time() + 145.0  # Safety margin
    rng = np.random.default_rng(42)
    
    # Initial configuration
    pts = _initial_fib()
    
    # Phase 1: Local Polish using exact hull metric
    # This directly attacks the bottleneck facets
    if time.time() < deadline - 2.0:
        pts = _local_polish(pts, deadline)
        
    # Final normalization
    pts = np.array(pts, dtype=float)
    pts /= (np.linalg.norm(pts, axis=1, keepdims=True) + _EPS)
    
    # Validity check
    if not np.all(np.isfinite(pts)) or pts.shape != (N, 3):
        return _initial_fib()
        
    return pts
