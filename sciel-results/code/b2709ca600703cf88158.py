import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

# Precompute face-to-point mapping for efficiency
_face_to_point = None

def _build_face_point_map(faces):
    """Build a mapping from point index to list of face indices containing it."""
    global _face_to_point
    _face_to_point = [[] for _ in range(N)]
    for fi, face in enumerate(faces):
        for pi in face:
            _face_to_point[pi].append(fi)
    return _face_to_point

def _fibonacci_spiral(n, seed=0.0):
    """Generate n points on the unit sphere using Fibonacci spiral with optional offset."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + seed
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _compute_facet_circumradii(pts):
    """
    Compute circumradius for each facet of the convex hull.
    Returns (radii, simplices, circumcenters) or (None, None, None) on failure.
    """
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        a = pts[faces[:, 0]]
        b = pts[faces[:, 1]]
        c = pts[faces[:, 2]]
        
        # Circumcenter direction for a spherical triangle:
        # c_dir = cross(a,b) + cross(b,c) + cross(c,a)
        ab = np.cross(a, b)
        bc = np.cross(b, c)
        ca = np.cross(c, a)
        c_dir = ab + bc + ca
        norms = np.linalg.norm(c_dir, axis=1)
        
        radii = np.zeros(len(faces))
        centers = np.zeros((len(faces), 3))
        valid = norms > 1e-10
        
        if np.any(valid):
            c_dir_valid = c_dir[valid] / norms[valid, np.newaxis]
            centers[valid] = c_dir_valid
            cos_r = np.clip(np.sum(c_dir_valid * a[valid], axis=1), -1.0, 1.0)
            r = np.arccos(cos_r)
            # Ensure we get the smaller angle (cap radius, not the reflex)
            r[r > np.pi / 2] = np.pi - r[r > np.pi / 2]
            radii[valid] = r
        
        if np.any(~valid):
            # Degenerate case: use max edge length
            a_deg = a[~valid]
            b_deg = b[~valid]
            c_deg = c[~valid]
            d_ab = np.arccos(np.clip(np.sum(a_deg * b_deg, axis=1), -1, 1))
            d_bc = np.arccos(np.clip(np.sum(b_deg * c_deg, axis=1), -1, 1))
            d_ca = np.arccos(np.clip(np.sum(c_deg * a_deg, axis=1), -1, 1))
            radii[~valid] = np.maximum(np.maximum(d_ab, d_bc), d_ca)
            # For degenerate, center is just the normalized sum
            s = a_deg + b_deg + c_deg
            sn = np.linalg.norm(s, axis=1, keepdims=True)
            sn[sn < 1e-10] = 1e-10
            centers[~valid] = s / sn
        
        return radii, faces, centers
    except Exception:
        return None, None, None

def _covering_radius(pts):
    """Compute the covering radius (max facet circumradius) in radians."""
    radii, _, _ = _compute_facet_circumradii(pts)
    if radii is None:
        return np.pi
    return float(np.max(radii))

def _repulsion_optimize(pts, alpha_init=0.02, decay=0.995, max_iter=500, deadline=None):
    """Optimize points using repulsion force to spread them evenly."""
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute pairwise dot products
        dot = np.dot(pts, pts.T)
        dot = np.clip(dot, -1.0, 1.0)
        np.fill_diagonal(dot, 1.0)
        dist_sq = 2.0 - 2.0 * dot
        dist_sq[dist_sq < 1e-10] = 1e-10
        
        # Repulsion: force ~ 1/d^2, gradient ~ -2/d^3 * (pts_i - pts_j)
        # Sum over j of (pts_i - pts_j) / |pts_i - pts_j|^3
        # = pts_i * sum(1/d^3) - sum(pts_j / d^3)
        inv_d3 = 1.0 / (dist_sq * np.sqrt(dist_sq))
        np.fill_diagonal(inv_d3, 0.0)
        
        # grad_i = sum_j (pts_i - pts_j) / |pts_i - pts_j|^3
        #        = pts_i * sum_j(1/d^3) - sum_j(pts_j / d^3)
        sum_inv_d3 = np.sum(inv_d3, axis=1, keepdims=True)
        sum_pts_inv_d3 = inv_d3 @ pts
        
        grad = pts * sum_inv_d3 - sum_pts_inv_d3
        
        # Project to tangent plane (remove radial component)
        radial = np.sum(grad * pts, axis=1, keepdims=True)
        grad = grad - radial * pts
        
        # Normalize and update
        grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
        grad_norm[grad_norm < 1e-10] = 1.0
        grad = grad / grad_norm
        
        pts = pts + alpha_init * grad
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        
        alpha_init *= decay
    
    return pts

def _facet_optimize(pts, lr_init=0.05, decay=0.995, max_iter=2000, deadline=None):
    """
    Optimize by moving each point toward the circumcenter of its worst facet.
    Uses vectorized operations for speed.
    """
    best_pts = pts.copy()
    best_radius = _covering_radius(pts)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        radii, faces, centers = _compute_facet_circumradii(pts)
        if radii is None:
            break
        
        # Build face-to-point mapping
        face_to_point = _build_face_point_map(faces)
        
        # For each point, find the worst facet containing it (vectorized)
        grad = np.zeros_like(pts)
        
        for j in range(N):
            facet_indices = face_to_point[j]
            if not facet_indices:
                continue
            
            worst_idx = facet_indices[np.argmax(radii[facet_indices])]
            worst_r = radii[worst_idx]
            
            if worst_r < 1e-10:
                continue
            
            # Move toward the circumcenter of the worst facet
            center = centers[worst_idx]
            # Tangent direction
            tangent = center - np.dot(center, pts[j]) * pts[j]
            tn = np.linalg.norm(tangent)
            if tn > 1e-10:
                tangent = tangent / tn
                grad[j] = tangent * min(worst_r, 1.0)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
        grad_norm[grad_norm < 1e-10] = 1.0
        grad = grad / grad_norm
        
        # Update
        pts = pts + lr_init * grad
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        
        # Check improvement
        current_radius = _covering_radius(pts)
        if current_radius < best_radius:
            best_radius = current_radius
            best_pts = pts.copy()
        
        lr_init *= decay
    
    return best_pts, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s kill
    
    best_pts = None
    best_radius = np.pi
    
    # Multiple restarts with different initializations
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        
        # Different initializations
        if restart == 0:
            pts = _fibonacci_spiral(N, seed=0.0)
        elif restart == 1:
            pts = _fibonacci_spiral(N, seed=0.5)
        elif restart == 2:
            pts = _fibonacci_spiral(N, seed=1.0)
        elif restart == 3:
            # Random initialization
            rng = np.random.RandomState(42 + restart)
            pts = rng.randn(N, 3)
            pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        else:
            # Perturbed best so far
            if best_pts is not None:
                rng = np.random.RandomState(100 + restart)
                pts = best_pts + 0.1 * rng.randn(N, 3)
                pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
            else:
                pts = _fibonacci_spiral(N, seed=float(restart))
        
        # Phase 1: Repulsion to get a good spread
        rep_deadline = deadline - 50.0
        if time.time() < rep_deadline:
            pts = _repulsion_optimize(pts, alpha_init=0.02, decay=0.995, 
                                       max_iter=300, deadline=rep_deadline)
        
        # Phase 2: Facet-based optimization
        facet_deadline = deadline - 15.0
        if time.time() < facet_deadline:
            pts, radius = _facet_optimize(pts, lr_init=0.05, decay=0.995,
                                           max_iter=1500, deadline=facet_deadline)
            
            if radius < best_radius:
                best_radius = radius
                best_pts = pts.copy()
        
        # Phase 3: Fine-tuning with smaller steps
        if best_pts is not None and time.time() < deadline - 5.0:
            fine_deadline = deadline - 5.0
            pts_fine, radius_fine = _facet_optimize(best_pts, lr_init=0.01, decay=0.998,
                                                      max_iter=1000, deadline=fine_deadline)
            if radius_fine < best_radius:
                best_radius = radius_fine
                best_pts = pts_fine.copy()
    
    # Final validation
    if best_pts is None:
        # Fallback: return Fibonacci spiral
        best_pts = _fibonacci_spiral(N, seed=0.0)
    
    # Ensure proper normalization
    best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    # Final check
    final_radius = _covering_radius(best_pts)
    if final_radius < best_radius:
        best_radius = final_radius
    
    return best_pts
