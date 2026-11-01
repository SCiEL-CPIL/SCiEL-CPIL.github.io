import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 50

def _spherical_circumcenters(hull, points):
    """
    Compute circumcenters and radii for all facets of the convex hull.
    Returns:
        centers: (M, 3) array of circumcenters (unit vectors)
        radii: (M,) array of angular radii (radians)
    """
    facets = hull.simplices
    centers = np.zeros((len(facets), 3))
    radii = np.zeros(len(facets))
    
    for i, face in enumerate(facets):
        v0 = points[face[0]]
        v1 = points[face[1]]
        v2 = points[face[2]]
        
        n = np.cross(v1 - v0, v2 - v1)
        norm = np.linalg.norm(n)
        if norm < 1e-10:
            centers[i] = v0
            radii[i] = 0
            continue
        
        center = n / norm
        
        eq = hull.equations[i]
        hull_normal = eq[:3]
        
        if np.dot(center, hull_normal) < 0:
            center = -center
            
        centers[i] = center
        radii[i] = np.arccos(np.clip(np.dot(center, v0), -1.0, 1.0))
        
    return centers, radii

def _covering_radius(points):
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi
    centers, radii = _spherical_circumcenters(hull, points)
    return np.max(radii)

def _soft_max_radius(points, beta=50.0):
    """
    Smooth surrogate for max facet radius using log-sum-exp.
    Returns (soft_max_value, gradients w.r.t. points)
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi, np.zeros_like(points)
    
    centers, radii = _spherical_circumcenters(hull, points)
    
    # Log-sum-exp for soft max
    max_r = np.max(radii)
    # Shift for numerical stability
    shifted = radii - max_r
    weights = np.exp(beta * shifted)
    weights /= weights.sum()
    soft_val = max_r - np.log(weights.sum()) / beta
    
    # Gradient: d(soft_max)/d(point_j)
    grads = np.zeros_like(points)
    facets = hull.simplices
    
    for i in range(len(facets)):
        w = weights[i]
        if w < 1e-15:
            continue
        face = facets[i]
        center = centers[i]
        r = radii[i]
        v0 = points[face[0]]
        
        # d(r_i)/d(v_j) for j in face
        # r_i = arccos(center . v_j)
        # dr/dv = -center / sqrt(1 - (center.v)^2)
        dot_cv = np.dot(center, v0)
        sin_r = np.sqrt(max(1.0 - dot_cv * dot_cv, 1e-15))
        
        for j in face:
            vj = points[j]
            # The circumcenter depends on all three vertices, but for a first-order
            # approximation, we use the gradient of arccos(center . vj) w.r.t. vj
            # holding center fixed (this is an approximation)
            dot_cj = np.dot(center, vj)
            sin_rj = np.sqrt(max(1.0 - dot_cj * dot_cj, 1e-15))
            grads[j] += w * (-center / sin_rj)
    
    return soft_val, grads

def _optimize_lbfgs(points, timeout=20.0):
    """
    Optimize using L-BFGS-B on the soft-max surrogate.
    """
    start_time = time.time()
    points = points.copy().flatten()
    
    def objective(x):
        if time.time() - start_time > timeout:
            raise Exception("timeout")
        pts = x.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1
        pts /= norms
        val, _ = _soft_max_radius(pts, beta=50.0)
        return val
    
    def gradient(x):
        if time.time() - start_time > timeout:
            raise Exception("timeout")
        pts = x.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1
        pts /= norms
        _, grads = _soft_max_radius(pts, beta=50.0)
        # Project gradient onto tangent space of sphere
        pts_flat = pts.flatten()
        grads_flat = grads.flatten()
        # Tangent projection: g - (g . p) * p for each point
        proj_grads = np.zeros_like(pts)
        for i in range(N):
            pi = pts[i]
            gi = grads[i]
            proj_grads[i] = gi - np.dot(gi, pi) * pi
        return proj_grads.flatten()
    
    try:
        result = minimize(
            objective, points, jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 500}
        )
        if time.time() - start_time > timeout:
            # Return current best if timeout during optimization
            pass
        opt_pts = result.x.reshape(N, 3)
    except Exception:
        opt_pts = points.reshape(N, 3)
    
    # Normalize
    norms = np.linalg.norm(opt_pts, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1
    opt_pts /= norms
    
    return opt_pts, _covering_radius(opt_pts)

def _local_polish(points, timeout=5.0):
    """
    Final polish: targeted moves of worst facet vertices toward circumcenters.
    """
    start_time = time.time()
    points = points.copy()
    
    for _ in range(50):
        if time.time() - start_time > timeout:
            break
        
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        centers, radii = _spherical_circumcenters(hull, points)
        max_r = np.max(radii)
        
        if max_r < 1e-8:
            break
        
        # Find worst facet
        worst_idx = np.argmax(radii)
        face = hull.simplices[worst_idx]
        center = centers[worst_idx]
        
        # Move vertices of worst facet toward center
        step = 0.01 * (1.0 - 0.9 * _ / 50.0)
        for j in face:
            v = points[j]
            dir_vec = center - v
            norm_dir = np.linalg.norm(dir_vec)
            if norm_dir > 1e-10:
                points[j] += step * dir_vec / norm_dir
        
        # Renormalize
        points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    return points, _covering_radius(points)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    best_points = None
    best_radius = np.pi
    
    # 1. Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    init_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    starts = [init_fib]
    
    # 2. Random starts
    rng = np.random.default_rng(42)
    for _ in range(15):
        if time.time() > deadline - 10:
            break
        p = rng.normal(size=(N, 3))
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        starts.append(p)
        
    for start in starts:
        if time.time() > deadline - 10:
            break
            
        remaining = deadline - time.time()
        if remaining < 15:
            break
            
        # Phase 1: L-BFGS on soft-max surrogate
        lbfgs_timeout = min(remaining - 10, 30.0)
        opt_pts, radius = _optimize_lbfgs(start, timeout=lbfgs_timeout)
        
        # Phase 2: Local polish
        if time.time() < deadline - 5:
            polish_timeout = min(deadline - time.time() - 3, 5.0)
            opt_pts, radius = _local_polish(opt_pts, timeout=polish_timeout)
        
        if radius < best_radius:
            best_radius = radius
            best_points = opt_pts.copy()
            
    if best_points is None:
        return init_fib
        
    return best_points
