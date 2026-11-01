import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

def _fibonacci_spiral(n, seed=0.0):
    """Generate n points on the unit sphere using Fibonacci spiral with optional offset."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + seed
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _facet_data(pts):
    """
    Compute convex hull facets and their circumradii.
    Returns (radii, faces, centers) or (None, None, None) on failure.
    """
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        a = pts[faces[:, 0]]
        b = pts[faces[:, 1]]
        c = pts[faces[:, 2]]
        
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
            r[r > np.pi / 2] = np.pi - r[r > np.pi / 2]
            radii[valid] = r
        
        if np.any(~valid):
            a_deg = a[~valid]
            b_deg = b[~valid]
            c_deg = c[~valid]
            d_ab = np.arccos(np.clip(np.sum(a_deg * b_deg, axis=1), -1, 1))
            d_bc = np.arccos(np.clip(np.sum(b_deg * c_deg, axis=1), -1, 1))
            d_ca = np.arccos(np.clip(np.sum(c_deg * a_deg, axis=1), -1, 1))
            radii[~valid] = np.maximum(np.maximum(d_ab, d_bc), d_ca)
            s = a_deg + b_deg + c_deg
            sn = np.linalg.norm(s, axis=1, keepdims=True)
            sn[sn < 1e-10] = 1e-10
            centers[~valid] = s / sn
        
        return radii, faces, centers
    except Exception:
        return None, None, None

def _covering_radius(pts):
    """Compute the covering radius in radians."""
    radii, _, _ = _facet_data(pts)
    if radii is None:
        return np.pi
    return float(np.max(radii))

def _repulsion_optimize(pts, alpha_init=0.03, decay=0.997, max_iter=400, deadline=None):
    """Optimize points using repulsion force to spread them evenly."""
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        dot = np.dot(pts, pts.T)
        dot = np.clip(dot, -1.0, 1.0)
        np.fill_diagonal(dot, 1.0)
        dist_sq = 2.0 - 2.0 * dot
        dist_sq[dist_sq < 1e-10] = 1e-10
        
        inv_d3 = 1.0 / (dist_sq * np.sqrt(dist_sq))
        np.fill_diagonal(inv_d3, 0.0)
        
        sum_inv_d3 = np.sum(inv_d3, axis=1, keepdims=True)
        sum_pts_inv_d3 = inv_d3 @ pts
        
        grad = pts * sum_inv_d3 - sum_pts_inv_d3
        
        radial = np.sum(grad * pts, axis=1, keepdims=True)
        grad = grad - radial * pts
        
        grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
        grad_norm[grad_norm < 1e-10] = 1.0
        grad = grad / grad_norm
        
        pts = pts + alpha_init * grad
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        
        alpha_init *= decay
    
    return pts

def _smooth_max(radii, beta=50.0):
    """
    Smooth maximum using log-sum-exp trick.
    Returns (smooth_max_value, weights) where weights sum to 1.
    """
    # Use a stable log-sum-exp
    m = np.max(radii)
    if m < -1e15:
        return 0.0, np.zeros_like(radii)
    shifted = radii - m
    exp_shifted = np.exp(beta * shifted)
    log_sum = np.log(np.sum(exp_shifted))
    smooth_val = m + np.log(np.sum(exp_shifted)) / beta
    weights = exp_shifted / np.sum(exp_shifted)
    return smooth_val, weights

def _facet_optimize_smooth(pts, beta_init=30.0, beta_max=200.0, lr_init=0.04, 
                           decay=0.998, max_iter=3000, deadline=None):
    """
    Optimize using smooth max of facet circumradii.
    Moves points toward circumcenters weighted by facet importance.
    """
    best_pts = pts.copy()
    best_radius = _covering_radius(pts)
    beta = beta_init
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        radii, faces, centers = _facet_data(pts)
        if radii is None:
            break
        
        # Smooth max and weights
        smooth_val, weights = _smooth_max(radii, beta)
        
        # For each point, compute gradient: weighted sum of directions to circumcenters
        grad = np.zeros_like(pts)
        for j in range(N):
            mask = np.any(faces == j, axis=1)
            if not np.any(mask):
                continue
            
            facet_indices = np.where(mask)[0]
            w = weights[facet_indices]
            
            # Weighted circumcenter direction
            weighted_center = np.sum(w[:, np.newaxis] * centers[facet_indices], axis=0)
            wc_norm = np.linalg.norm(weighted_center)
            if wc_norm > 1e-10:
                weighted_center = weighted_center / wc_norm
            
            # Tangent direction toward weighted center
            tangent = weighted_center - np.dot(weighted_center, pts[j]) * pts[j]
            tn = np.linalg.norm(tangent)
            if tn > 1e-10:
                grad[j] = tangent / tn
        
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
        
        # Increase beta to sharpen the max
        beta = min(beta * 1.02, beta_max)
        lr_init *= decay
    
    return best_pts, best_radius

def _local_polish(pts, lr_init=0.005, max_iter=2000, deadline=None):
    """
    High-precision polish: move each point toward its worst facet circumcenter
    with very small steps.
    """
    best_pts = pts.copy()
    best_radius = _covering_radius(pts)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        radii, faces, centers = _facet_data(pts)
        if radii is None:
            break
        
        grad = np.zeros_like(pts)
        
        for j in range(N):
            mask = np.any(faces == j, axis=1)
            if not np.any(mask):
                continue
            
            facet_indices = np.where(mask)[0]
            worst_idx = facet_indices[np.argmax(radii[facet_indices])]
            worst_r = radii[worst_idx]
            
            if worst_r < 1e-10:
                continue
            
            center = centers[worst_idx]
            tangent = center - np.dot(center, pts[j]) * pts[j]
            tn = np.linalg.norm(tangent)
            if tn > 1e-10:
                grad[j] = (tangent / tn) * min(worst_r, 1.0)
        
        grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
        grad_norm[grad_norm < 1e-10] = 1.0
        grad = grad / grad_norm
        
        pts = pts + lr_init * grad
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        
        current_radius = _covering_radius(pts)
        if current_radius < best_radius:
            best_radius = current_radius
            best_pts = pts.copy()
        
        lr_init *= 0.999
    
    return best_pts, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    best_pts = None
    best_radius = np.pi
    
    # Phase 1: Multiple initializations with repulsion
    initializations = [
        _fibonacci_spiral(N, seed=0.0),
        _fibonacci_spiral(N, seed=0.3),
        _fibonacci_spiral(N, seed=0.7),
        _fibonacci_spiral(N, seed=1.1),
    ]
    
    for init_idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 80.0:
            break
        
        # Repulsion phase
        rep_deadline = deadline - 70.0
        if time.time() < rep_deadline:
            pts = _repulsion_optimize(init_pts, alpha_init=0.03, decay=0.997,
                                       max_iter=300, deadline=rep_deadline)
        else:
            pts = init_pts.copy()
        
        # Smooth facet optimization
        smooth_deadline = deadline - 30.0
        if time.time() < smooth_deadline:
            pts, radius = _facet_optimize_smooth(pts, beta_init=30.0, beta_max=150.0,
                                                  lr_init=0.04, decay=0.998,
                                                  max_iter=2000, deadline=smooth_deadline)
            
            if radius < best_radius:
                best_radius = radius
                best_pts = pts.copy()
        
        # Local polish
        if best_pts is not None and time.time() < deadline - 10.0:
            polish_deadline = deadline - 10.0
            pts_polish, radius_polish = _local_polish(best_pts, lr_init=0.005,
                                                       max_iter=1500, deadline=polish_deadline)
            if radius_polish < best_radius:
                best_radius = radius_polish
                best_pts = pts_polish.copy()
    
    # Final fine polish on best result
    if best_pts is not None and time.time() < deadline - 3.0:
        fine_deadline = deadline - 3.0
        pts_fine, radius_fine = _local_polish(best_pts, lr_init=0.002,
                                               max_iter=1000, deadline=fine_deadline)
        if radius_fine < best_radius:
            best_radius = radius_fine
            best_pts = pts_fine.copy()
    
    # Fallback
    if best_pts is None:
        best_pts = _fibonacci_spiral(N, seed=0.0)
    
    # Final normalization
    best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    return best_pts
