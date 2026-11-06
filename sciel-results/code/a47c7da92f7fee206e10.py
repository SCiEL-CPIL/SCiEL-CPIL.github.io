import numpy as np
import time
import sys
from scipy.spatial import ConvexHull
import math

N = 130

def _normalize(v):
    """Normalize vectors to unit length."""
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms

def _covering_radius_deg(pts):
    """
    Calculate the covering radius in degrees.
    The covering radius is the maximum distance from any point on the sphere
    to the nearest configuration point.
    This is equal to the maximum circumradius of the facets of the convex hull.
    """
    try:
        hull = ConvexHull(pts)
        facets = hull.simplices
        
        max_circumradius_deg = 0.0
        
        for facet in facets:
            p1, p2, p3 = pts[facet[0]], pts[facet[1]], pts[facet[2]]
            
            # Calculate circumcenter of the spherical triangle
            v1 = p1 - p2
            v2 = p2 - p3
            c_vec = np.cross(v1, v2)
            
            if np.linalg.norm(c_vec) < 1e-10:
                centroid = _normalize((p1+p2+p3)/3.0)
                d1 = np.arccos(np.clip(np.dot(centroid, p1), -1.0, 1.0))
                d2 = np.arccos(np.clip(np.dot(centroid, p2), -1.0, 1.0))
                d3 = np.arccos(np.clip(np.dot(centroid, p3), -1.0, 1.0))
                r_rad = np.max([d1, d2, d3])
            else:
                c_vec = _normalize(c_vec)
                d1 = np.arccos(np.clip(np.dot(c_vec, p1), -1.0, 1.0))
                d2 = np.arccos(np.clip(np.dot(-c_vec, p1), -1.0, 1.0))
                r_rad = d1 if d1 < d2 else d2
            
            r_deg = np.degrees(r_rad)
            if r_deg > max_circumradius_deg:
                max_circumradius_deg = r_deg
                
    except Exception:
        return 180.0
        
    return max_circumradius_deg

def _spectral_energy(pts):
    """
    Computes a spectral proxy for non-uniformity.
    We use the variance of the pair correlation function.
    For a set of N points, the pair correlation function g(r) measures
    how points are distributed relative to a uniform distribution.
    
    A simpler spectral approach:
    Minimize the sum of squared magnitudes of spherical harmonic coefficients
    of the empirical measure minus the uniform measure.
    However, computing full SH coefficients is expensive.
    
    Alternative Spectral Energy:
    Minimize the trace of the Gram matrix of the points (which is related to packing)
    or use a multi-scale repulsion.
    
    Let's use a "clumping" energy based on the inverse of the distance to the k-th nearest neighbor,
    but smoothed. Actually, the prompt asks for "spectral energy minimization... variance of pair correlation".
    
    Let's implement a simplified spectral uniformity measure:
    The energy E = sum_{l=1}^{L} sum_{m=-l}^{l} |Y_lm^* integral f dOmega|^2
    where f is the sum of delta functions.
    For a uniform distribution, these coefficients (except l=0) should be 0.
    We want to minimize this energy.
    
    To avoid heavy SH computation, we can use a random Fourier features approximation
    or just stick to a strong repulsion that is known to work well for covering:
    The "Thomson" energy (1/r) is good for packing, but for covering, we want to
    maximize the minimum distance to the nearest neighbor.
    
    Let's try a different spectral-like approach:
    Minimize the sum of exp(-alpha * dist_ij) for all pairs.
    This is a "soft" repulsion that penalizes close pairs.
    By tuning alpha, we can balance local and global uniformity.
    This is effectively minimizing a spectral measure of clustering.
    """
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, 1e-10)
    dist = np.sqrt(dist_sq)
    
    # Energy: sum of exp(-alpha * dist)
    # This penalizes points that are too close.
    # If alpha is large, it's like hard repulsion.
    # If alpha is small, it's like a global smoothing.
    alpha = 5.0
    energy = np.sum(np.exp(-alpha * dist))
    return energy

def _gradient_spectral(pts, alpha=5.0):
    """
    Compute gradient of the spectral energy w.r.t. points.
    E = sum_{i != j} exp(-alpha * d_ij)
    dE/dp_i = sum_{j != i} -exp(-alpha * d_ij) * (-alpha) * (p_i - p_j) / d_ij
             = alpha * sum_{j != i} exp(-alpha * d_ij) * (p_i - p_j) / d_ij
             
    This force pushes points apart.
    """
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, 1e-10)
    dist = np.sqrt(dist_sq)
    
    # weights = alpha * exp(-alpha * dist) / dist
    weights = alpha * np.exp(-alpha * dist) / dist
    np.fill_diagonal(weights, 0.0)
    
    # Gradient is sum_j weights_ij * (p_i - p_j)
    # This is equivalent to:
    # grad_i = sum_j weights_ij * p_i - sum_j weights_ij * p_j
    #        = p_i * sum_j weights_ij - sum_j (weights_ij * p_j)
    
    sum_weights = np.sum(weights, axis=1, keepdims=True)
    weighted_pts = np.sum(weights[:, :, np.newaxis] * pts[np.newaxis, :, :], axis=1)
    
    grad = sum_weights * pts - weighted_pts
    
    # Project to tangent plane
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad_tangent = grad - dot * pts
    
    return grad_tangent

def construct():
    start_time = time.time()
    time_limit = 145.0
    
    # 1. Initialization: Fibonacci spiral + slight random jitter
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)
    
    # Add small random perturbation to break symmetry
    noise = np.random.randn(N, 3) * 0.01
    dot = np.sum(noise * pts, axis=1, keepdims=True)
    noise_tangent = noise - dot * pts
    pts = _normalize(pts + noise_tangent)
    
    # 2. Spectral Energy Minimization (Gradient Descent)
    # We minimize the "clumping" energy which promotes uniformity.
    lr = 0.005
    alpha = 5.0
    max_iter = 2000
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit - 20:
            break
        
        grad = _gradient_spectral(pts, alpha)
        
        # Update
        pts = pts - lr * grad
        pts = _normalize(pts)
        
        # Adaptive learning rate: decrease slowly
        if it % 100 == 0:
            lr *= 0.95

    # 3. Local Repair: Target the worst facet
    # The spectral energy minimization gives a good global start,
    # but might leave a large hole. We now locally optimize to reduce the covering radius.
    
    best_score = _covering_radius_deg(pts)
    
    # Local search on the worst facet
    for _ in range(500):
        if time.time() - start_time > time_limit - 5:
            break
            
        # Find the facet with the largest circumradius
        try:
            hull = ConvexHull(pts)
            facets = hull.simplices
            max_r = 0.0
            worst_facet = None
            worst_circumcenter = None
            
            for facet in facets:
                p1, p2, p3 = pts[facet[0]], pts[facet[1]], pts[facet[2]]
                v1 = p1 - p2
                v2 = p2 - p3
                c_vec = np.cross(v1, v2)
                if np.linalg.norm(c_vec) < 1e-10:
                    continue
                c_vec = _normalize(c_vec)
                d1 = np.arccos(np.clip(np.dot(c_vec, p1), -1.0, 1.0))
                d2 = np.arccos(np.clip(np.dot(-c_vec, p1), -1.0, 1.0))
                r_rad = d1 if d1 < d2 else d2
                if r_rad > max_r:
                    max_r = r_rad
                    worst_facet = facet
                    worst_circumcenter = c_vec if d1 < d2 else -c_vec
                    
            if worst_facet is None:
                break
                
            # Move the points of the worst facet towards the circumcenter?
            # No, moving them towards the center reduces the circumradius?
            # Actually, to reduce the circumradius of a triangle, we should
            # expand the triangle or move the vertices away from the circumcenter
            # IF the circumcenter is inside. But here we want to cover the sphere.
            # The "hole" is at the circumcenter. We want to move points INTO the hole.
            # So we should move the vertices of the worst facet TOWARDS the circumcenter.
            
            # Let's try moving the 3 points of the worst facet towards the circumcenter
            # by a small amount.
            step = 0.05
            new_pts = pts.copy()
            for idx in worst_facet:
                # Move point towards circumcenter
                # Direction: (circumcenter - point)
                dir_vec = worst_circumcenter - new_pts[idx]
                dir_vec = _normalize(dir_vec)
                new_pts[idx] = new_pts[idx] + step * dir_vec
            
            new_pts = _normalize(new_pts)
            new_score = _covering_radius_deg(new_pts)
            
            if new_score < best_score:
                pts = new_pts
                best_score = new_score
            else:
                # If it didn't help, try moving them slightly apart to see if it opens up
                # or just accept the stagnation and try a random kick
                pass
                
        except Exception:
            break
            
    return _normalize(pts)

if __name__ == "__main__":
    pts = construct()
    print(f"Shape: {pts.shape}")
    print(f"Norms: {np.linalg.norm(pts, axis=1)[:5]}")
    print(f"Covering Radius: {_covering_radius_deg(pts)}")
