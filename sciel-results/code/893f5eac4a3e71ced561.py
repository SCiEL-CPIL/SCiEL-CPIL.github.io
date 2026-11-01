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

def _compute_hull_and_radii(pts):
    """
    Compute convex hull and angular radii of all facets.
    Returns:
        hull: ConvexHull object
        faces: array of facet indices
        normals: array of outward normals for valid facets
        ang_radii: array of angular radii for valid facets
        valid_mask: boolean mask for valid facets
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
        
        # Ensure normals point outward
        centroids = (A[valid] + B[valid] + C[valid]) / 3.0
        signs = np.sum(normals * centroids, axis=1, keepdims=True)
        normals = normals * np.sign(signs)
        normals = normals / (norms + _EPS)
        
        # Angular circumradius: arccos(normal . vertex)
        cos_theta = np.sum(normals * A[valid], axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        ang_radii = np.arccos(cos_theta)
        
        return hull, faces, normals, ang_radii, valid
    except Exception:
        return None, None, None, None, None

def _get_worst_facet_data(pts):
    """
    Get data for the facet with the largest covering radius.
    Returns:
        worst_idx: index of the worst facet in the filtered array
        v_indices: indices of the 3 vertices of this facet in the original pts array
        pole: the normal vector (circumcenter direction) of this facet
        current_radius: the angular radius of this facet
    """
    _, faces, normals, ang_radii, valid = _compute_hull_and_radii(pts)
    if faces is None:
        return None, None, None, np.pi
        
    worst_idx = np.argmax(ang_radii)
    v_indices = faces[valid][worst_idx]
    pole = normals[worst_idx]
    current_radius = ang_radii[worst_idx]
    
    return worst_idx, v_indices, pole, current_radius

def _move_vertices_towards_pole(pts, v_indices, pole, step_size):
    """
    Move the specified vertices towards the pole direction.
    Returns new points array.
    """
    new_pts = pts.copy()
    for v in v_indices:
        # Direction from current point to pole
        # Since both are on unit sphere, pole - pts[v] is the chord direction
        dir_vec = pole - new_pts[v]
        dir_norm = np.linalg.norm(dir_vec)
        if dir_norm > _EPS:
            dir_vec /= dir_norm
        else:
            continue
            
        # Move along this direction
        # We use a small step in the direction of the pole on the sphere
        # This is equivalent to rotating the point towards the pole
        # A simple linear interpolation on the sphere (slerp approximation for small steps)
        # new_pt = pts[v] + step * dir_vec
        # Then normalize
        
        new_pts[v] = new_pts[v] + step_size * dir_vec
        new_pts[v] /= (np.linalg.norm(new_pts[v]) + _EPS)
        
    return new_pts

def _random_perturb(pts, magnitude):
    """
    Add random noise to all points to escape local minima.
    """
    noise = np.random.standard_normal(pts.shape) * magnitude
    new_pts = pts + noise
    new_pts /= (np.linalg.norm(new_pts, axis=1, keepdims=True) + _EPS)
    return new_pts

def construct():
    deadline = time.time() + 145.0  # Safety margin for 150s limit
    rng = np.random.default_rng(42)
    
    # Start with Fibonacci initialization
    best_pts = _initial_fib()
    _, _, _, best_radii, _ = _compute_hull_and_radii(best_pts)
    if best_radii is not None:
        best_radius = np.max(best_radii)
    else:
        best_radius = np.pi
    
    # Simulated Annealing / Direct Descent Parameters
    step_size = 0.05
    perturb_prob = 0.1
    perturb_mag = 0.1
    cooling_rate = 0.999
    
    no_improve_count = 0
    max_no_improve = 100
    
    while time.time() < deadline:
        # Get current worst facet
        worst_idx, v_indices, pole, curr_radius = _get_worst_facet_data(best_pts)
        
        if v_indices is None:
            break
            
        # If we found a better radius, update best
        if curr_radius < best_radius:
            best_radius = curr_radius
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # If stuck for too long, increase perturbation or reset step
        if no_improve_count > max_no_improve:
            # Try a larger random kick
            kick_pts = _random_perturb(best_pts, 0.5)
            _, _, _, kick_radii, _ = _compute_hull_and_radii(kick_pts)
            if kick_radii is not None:
                kick_radius = np.max(kick_radii)
                if kick_radius < best_radius:
                    best_pts = kick_pts
                    best_radius = kick_radius
                    no_improve_count = 0
                    continue
            # If kick didn't help, reduce step size and increase perturbation probability
            step_size *= 0.9
            perturb_prob = min(0.5, perturb_prob * 1.1)
            perturb_mag = min(0.5, perturb_mag * 1.05)
            no_improve_count = 0
            if step_size < 1e-5:
                break
        
        # Decide whether to perturb or do direct descent
        if rng.random() < perturb_prob:
            cand_pts = _random_perturb(best_pts, perturb_mag)
        else:
            cand_pts = _move_vertices_towards_pole(best_pts, v_indices, pole, step_size)
        
        # Evaluate candidate
        cand_hull, cand_faces, cand_normals, cand_radii, cand_valid = _compute_hull_and_radii(cand_pts)
        if cand_radii is None:
            continue
            
        cand_radius = np.max(cand_radii)
        
        # Accept if better
        if cand_radius < best_radius:
            best_pts = cand_pts
            best_radius = cand_radius
            # Slightly increase step size if we made progress? Or keep it stable.
            # Let's keep step size stable for direct descent, but cool down perturbation
            perturb_mag *= 0.99
            step_size = min(0.1, step_size * 1.01) # Very slight increase to allow larger moves if possible
        else:
            # Reject, but cool down
            step_size *= cooling_rate
            perturb_mag *= cooling_rate

    # Final normalization and validity check
    best_pts = np.array(best_pts, dtype=float)
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts /= (norms + _EPS)
    
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (N, 3):
        return _initial_fib()
        
    return best_pts
