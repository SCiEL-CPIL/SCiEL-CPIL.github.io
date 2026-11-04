"""
Sphere covering with 130 points using a combined force-directed + gradient-based approach.

Strategy:
1. Vectorized Coulomb repulsion for global uniformity (fast, no hull).
2. Gradient-based refinement using soft-max over facet circumradii (precise local search).
3. Analytic gradients for arccos-based circumradii to enable efficient descent.
"""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130
_DEADLINE = time.time() + 145.0  # 5s safety margin

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _time_ok(margin=5.0):
    return time.time() < _DEADLINE - margin

def _facet_circumradii(points, simplices):
    """
    Compute circumradii and circumcenters for all hull facets (vectorized).
    Returns:
        radii: (M,) array of angular circumradii
        centers: (M, 3) array of circumcenters on sphere
        mask: (M,) boolean array of valid facets
    """
    M = len(simplices)
    if M == 0:
        return np.array([]), np.array([]).reshape(0, 3), np.array([], dtype=bool)
    
    p1 = points[simplices[:, 0]]
    p2 = points[simplices[:, 1]]
    p3 = points[simplices[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = np.cross(v1, v2)
    norm_cross = np.linalg.norm(cross, axis=1)
    
    valid = norm_cross > 1e-14
    centers = np.zeros((M, 3))
    radii = np.zeros(M)
    
    if np.any(valid):
        c = cross[valid] / norm_cross[valid, np.newaxis]
        
        centroid = (p1[valid] + p2[valid] + p3[valid]) / 3.0
        dot_c_centroid = np.sum(c * centroid, axis=1)
        flip = dot_c_centroid < 0
        c[flip] = -c[flip]
        
        centers[valid] = c
        dot_val = np.clip(np.sum(c * p1[valid], axis=1), -1.0, 1.0)
        dist = np.arccos(dot_val)
        radii[valid] = dist
    
    return radii, centers, valid

def _covering_radius(pts):
    """Compute the covering radius in radians via convex hull facet circumcenters."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi / 2
    
    radii, centers, valid = _facet_circumradii(pts, hull.simplices)
    if not np.any(valid):
        return np.pi / 2
    return float(np.max(radii[valid]))

def _fibonacci_spiral(n, jitter=0.0, seed=0):
    """Fibonacci spiral on the unit sphere with optional jitter."""
    rng = np.random.RandomState(seed)
    i = np.arange(n) + 0.5
    if jitter > 0:
        i = i + rng.randn(n) * jitter
        i = np.clip(i, 0.5, n - 0.5)
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)

def _coulomb_force_step(pts, lr=0.01):
    """
    Vectorized Coulomb repulsion step.
    F_i = sum_j (pts[i] - pts[j]) / |pts[i] - pts[j]|^3
    """
    pts = _normalize(pts.copy())
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist3_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist3_sq, 1.0)
    dist3 = np.sqrt(dist3_sq)
    
    inv_dist3 = 1.0 / (dist3 ** 3)
    inv_dist3[dist3 < 1e-6] = 0.0
    F = (diff * inv_dist3[:, :, np.newaxis]).sum(axis=1)
    
    # Project to tangent plane
    F = F - np.einsum('ij,ij->i', F, pts)[:, np.newaxis] * pts
    
    F_norms = np.linalg.norm(F, axis=1, keepdims=True)
    F_norms[F_norms < 1e-12] = 1.0
    F = F / F_norms
    
    new_pts = pts + lr * F
    new_pts = _normalize(new_pts)
    
    return new_pts

def _gradient_step(pts, beta=50.0, lr=0.01):
    """
    Gradient descent step using soft-max over facet circumradii.
    Returns new points and the objective value.
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return pts, _covering_radius(pts)
    
    radii, centers, valid = _facet_circumradii(pts, hull.simplices)
    M = len(radii)
    
    if M == 0 or not np.any(valid):
        return pts, 0.0
    
    r_valid = radii[valid]
    c_valid = centers[valid]
    f_valid = hull.simplices[valid]
    
    if len(r_valid) == 0:
        return pts, 0.0
    
    r_max = np.max(r_valid)
    if r_max < 1e-15:
        return pts, 0.0
    
    shifted = beta * (r_valid - r_max)
    log_sum_exp = r_max + np.log(np.sum(np.exp(shifted))) / beta
    
    weights = np.exp(shifted) / np.sum(np.exp(shifted))
    
    grad = np.zeros_like(pts)
    
    for k in range(len(f_valid)):
        if weights[k] < 1e-10:
            continue
        
        tri = f_valid[k]
        c = c_valid[k]
        r_k = r_valid[k]
        
        if r_k < 1e-12:
            continue
        
        for idx, p in zip(tri, [pts[tri[0]], pts[tri[1]], pts[tri[2]]]):
            dot_cp = np.dot(c, p)
            sin_r = np.sqrt(max(0.0, 1.0 - dot_cp * dot_cp))
            if sin_r > 1e-12:
                # Gradient of arccos(c.p) w.r.t. p: -(c - (c.p)p) / sin(r)
                grad_p = -(c - dot_cp * p) / sin_r
                # Project to tangent plane
                grad_p -= np.dot(grad_p, p) * p
                grad[idx] += weights[k] * grad_p
    
    # Normalize gradient for stable steps
    grad_norms = np.linalg.norm(grad, axis=1, keepdims=True)
    grad_norms[grad_norms < 1e-12] = 1.0
    grad = grad / grad_norms
    
    new_pts = pts - lr * grad
    new_pts = _normalize(new_pts)
    
    return new_pts, float(log_sum_exp)

def construct():
    """
    1. Generate diverse initial configs.
    2. Run fast Coulomb force-directed optimization (no hull).
    3. Refine the best candidate with gradient-based local search.
    """
    start_time = time.time()
    best_pts = None
    best_r = np.pi
    
    # Initial configs
    inits = []
    for seed in [42, 123, 7, 999, 31415, 271828, 161803, 111, 222, 333]:
        inits.append(_fibonacci_spiral(N, jitter=0.05, seed=seed))
    inits.append(_fibonacci_spiral(N, jitter=0.0, seed=0))
    rng = np.random.RandomState(555)
    inits.append(_normalize(rng.randn(N, 3)))
    
    # Sort by initial covering radius
    init_scores = []
    for p in inits:
        try:
            r = _covering_radius(p)
        except Exception:
            r = np.pi
        init_scores.append((r, p))
    init_scores.sort(key=lambda x: x[0])
    
    # Phase 1: Fast Coulomb Optimization
    for r0, p0 in init_scores:
        if not _time_ok(15.0):
            break
            
        pts = p0.copy()
        prev_cr = _covering_radius(pts)
        
        for it in range(300):
            if not _time_ok(10.0):
                break
            pts = _coulomb_force_step(pts, lr=0.008)
            
            if (it + 1) % 50 == 0:
                cr_check = _covering_radius(pts)
                if abs(cr_check - prev_cr) < 1e-5:
                    break
                prev_cr = cr_check
        
        cr = _covering_radius(pts)
        if cr < best_r:
            best_r = cr
            best_pts = pts.copy()
    
    # Phase 2: Gradient-based Refinement
    if best_pts is not None and _time_ok(10.0):
        pts = best_pts.copy()
        lr = 0.015
        beta = 50.0
        stall_count = 0
        
        for it in range(150):
            if not _time_ok(5.0):
                break
            
            new_pts, obj = _gradient_step(pts, beta=beta, lr=lr)
            
            cr = _covering_radius(new_pts)
            if cr < best_r:
                best_r = cr
                best_pts = new_pts.copy()
                pts = new_pts
                stall_count = 0
            else:
                stall_count += 1
                lr *= 0.85
                if lr < 0.001:
                    break
            
            if stall_count > 15:
                break
            
            beta *= 1.02  # Sharpen the soft-max
    
    # Phase 3: Fine polish
    if best_pts is not None and _time_ok(3.0):
        pts = best_pts.copy()
        lr = 0.003
        for it in range(50):
            if not _time_ok(1.5):
                break
            
            new_pts, obj = _gradient_step(pts, beta=100.0, lr=lr)
            cr = _covering_radius(new_pts)
            
            if cr < best_r:
                best_r = cr
                best_pts = new_pts.copy()
                pts = new_pts
            else:
                lr *= 0.7
                if lr < 0.0003:
                    break
    
    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
    
    return _normalize(best_pts)

# Test
if __name__ == "__main__":
    t0 = time.time()
    pts = construct()
    print("Shape:", pts.shape)
    print("Covering radius (deg):", np.degrees(_covering_radius(pts)))
    print("Time: %.1f s" % (time.time() - t0))
