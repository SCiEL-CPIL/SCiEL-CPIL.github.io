import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130
START_TIME = time.time()
TIME_LIMIT = 110.0  # safety margin before 120s

def _fibonacci_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def _covering_radius_surrogate(pts, sample_pts, beta=50.0):
    """
    Compute a smooth surrogate for the covering radius using log-sum-exp.
    pts: (N, 3) configuration points
    sample_pts: (M, 3) sample points on sphere
    Returns: soft-max of min distances
    """
    # Compute all pairwise distances (angular)
    # For efficiency, compute dot products
    dots = sample_pts @ pts.T  # (M, N)
    # Clip for numerical stability
    dots = np.clip(dots, -1.0, 1.0)
    # Angular distance
    angles = np.arccos(dots)  # (M, N)
    # Min distance from each sample point to nearest config point
    min_angles = np.min(angles, axis=1)  # (M,)
    # Soft-max (log-sum-exp) as smooth approximation of max
    # Use scaled version for numerical stability
    scaled = beta * min_angles
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_max = log_sum_exp / beta
    return soft_max

def _covering_radius_surrogate_grad(pts, sample_pts, beta=50.0):
    """
    Compute gradient of the soft-max surrogate.
    """
    dots = sample_pts @ pts.T  # (M, N)
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)  # (M, N)
    min_angles = np.min(angles, axis=1)  # (M,)
    
    # Soft-max weights
    scaled = beta * min_angles
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    sum_exp = np.sum(exp_scaled)
    weights = exp_scaled / sum_exp  # (M,)
    
    # Gradient: d(soft_max)/d(pts[j]) = sum_i weights[i] * d(min_angle_i)/d(pts[j])
    # min_angle_i = min_j angle(i, j)
    # d(angle)/d(pts[j]) = -pts[i] / sqrt(1 - (pts[i]·pts[j])^2) + pts[j] * (pts[i]·pts[j]) / sqrt(1 - (pts[i]·pts[j])^2)^2 * ... 
    # Actually, angle = arccos(dot), d(angle)/d(dot) = -1/sqrt(1-dot^2)
    # d(dot)/d(pts[j]) = pts[i]
    # So d(angle)/d(pts[j]) = -pts[i] / sqrt(1 - dot^2)
    # But only for the j that achieves the min
    
    # For the soft version, we need to be more careful. Let's use the fact that
    # min_angle_i is achieved at argmin_j angle(i,j). For the soft version, we approximate.
    
    # Simpler: use the subgradient of the min, weighted by the soft-max weights
    # This is an approximation but should work for optimization
    
    # For each sample point i, find the nearest config point
    nearest_idx = np.argmin(angles, axis=1)  # (M,)
    
    # Gradient contribution
    grad = np.zeros_like(pts)
    for i in range(len(sample_pts)):
        j = nearest_idx[i]
        w = weights[i]
        dot = dots[i, j]
        # d(angle)/d(pts[j]) = -pts[i] / sqrt(1 - dot^2)
        denom = np.sqrt(max(1e-12, 1.0 - dot * dot))
        grad[j] += w * (-sample_pts[i] / denom)
    
    return soft_max_val, grad

def _covering_radius_exact(pts):
    """Compute the exact covering radius using convex hull."""
    hull = ConvexHull(pts)
    max_radius = 0.0
    for simplex in hull.simplices:
        # Get the 3 points of the facet
        facet_pts = pts[simplex]
        # Compute spherical circumradius
        # The circumcenter is the point on the sphere equidistant from all 3 points
        # In 3D, the circumcenter of a triangle on the unit sphere
        a, b, c = facet_pts
        # Circumcenter: solve |x-a|^2 = |x-b|^2 = |x-c|^2 = r^2, |x|=1
        # This is equivalent to: x·(a-b) = (|a|^2 - |b|^2)/2 = 0 (since |a|=|b|=1)
        # So x is perpendicular to (a-b) and (a-c)
        # x is in the direction of (a-b) × (a-c) normalized... no, that's the normal
        # Actually, the circumcenter lies in the plane of the triangle
        # Let's use the formula: circumradius R = abc / (4*Area) for planar triangle
        # But for spherical, we need the angular circumradius
        # The angular circumradius is the angle from the circumcenter to each vertex
        # Compute using the formula for spherical triangle circumradius
        
        # Alternative: compute the circumcenter in 3D
        # The circumcenter of triangle abc on unit sphere
        # It's the intersection of the perpendicular bisectors
        # Perpendicular bisector of ab: points x with x·a = x·b, i.e., x·(a-b) = 0
        # So circumcenter is in the plane through origin perpendicular to (a-b)
        # And also perpendicular to (a-c)
        # So circumcenter is along (a-b) × (a-c) direction? No, that's the normal to the plane
        # Wait, the circumcenter is in the plane of the triangle (span of a, b, c)
        # Let's use: circumcenter = alpha*a + beta*b + gamma*c with constraints
        
        # Simpler numerical approach: 
        # The circumcenter x satisfies: x·a = x·b = x·c = cos(R)
        # And |x| = 1
        # From x·a = x·b: x·(a-b) = 0
        # From x·a = x·c: x·(a-c) = 0
        # So x is perpendicular to (a-b) and (a-c)
        # Thus x is parallel to (a-b) × (a-c)
        cross = np.cross(a - b, a - c)
        norm_cross = np.linalg.norm(cross)
        if norm_cross < 1e-12:
            continue
        x = cross / norm_cross
        # x should be unit, and x·a = cos(R)
        cos_R = x @ a
        cos_R = np.clip(cos_R, -1.0, 1.0)
        R = np.arccos(cos_R)
        if R > max_radius:
            max_radius = R
    return np.degrees(max_radius)

def construct():
    global START_TIME
    START_TIME = time.time()
    
    # Initial configuration
    pts = _fibonacci_spiral(N).astype(np.float64)
    pts = _normalize(pts)
    
    # Create sample points for surrogate
    # Use a moderately dense set
    n_sample = 20000
    # Generate sample points using Fibonacci spiral with more points
    sample_pts = _fibonacci_spiral(n_sample)
    sample_pts = _normalize(sample_pts)
    
    best_pts = pts.copy()
    best_score = _covering_radius_exact(pts)
    
    # Optimization using L-BFGS-B with smooth surrogate
    beta = 30.0  # soft-max parameter
    
    def objective_and_grad(flat_pts):
        p = flat_pts.reshape(N, 3)
        p = _normalize(p)
        
        # Compute soft-max surrogate
        dots = sample_pts @ p.T  # (M, N)
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(dots)  # (M, N)
        min_angles = np.min(angles, axis=1)  # (M,)
        
        scaled = beta * min_angles
        max_scaled = np.max(scaled)
        exp_scaled = np.exp(scaled - max_scaled)
        sum_exp = np.sum(exp_scaled)
        weights = exp_scaled / sum_exp  # (M,)
        soft_max = (max_scaled + np.log(sum_exp)) / beta
        
        # Gradient
        nearest_idx = np.argmin(angles, axis=1)
        grad = np.zeros_like(p)
        for i in range(len(sample_pts)):
            j = nearest_idx[i]
            w = weights[i]
            dot = dots[i, j]
            denom = np.sqrt(max(1e-12, 1.0 - dot * dot))
            grad[j] += w * (-sample_pts[i] / denom)
        
        # Project gradient to tangent space (remove radial component)
        for j in range(N):
            grad[j] -= np.dot(grad[j], p[j]) * p[j]
        
        return soft_max, grad.flatten()
    
    # Use custom optimization loop with gradient descent
    lr = 0.01
    max_iter = 2000
    
    for it in range(max_iter):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        
        val, grad = objective_and_grad(pts.flatten())
        
        # Gradient descent step
        pts_flat = pts.flatten() - lr * grad
        pts = _normalize(pts_flat.reshape(N, 3))
        
        # Periodically check exact covering radius
        if it % 100 == 0 or it == max_iter - 1:
            exact_score = _covering_radius_exact(pts)
            if exact_score < best_score:
                best_score = exact_score
                best_pts = pts.copy()
        
        # Adaptive learning rate
        if it % 200 == 0 and it > 0:
            lr *= 0.9
    
    # Final check
    exact_score = _covering_radius_exact(pts)
    if exact_score < best_score:
        best_score = exact_score
        best_pts = pts.copy()
    
    return best_pts
