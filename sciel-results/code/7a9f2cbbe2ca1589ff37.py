import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

def _covering_radius(points):
    """
    Computes the covering radius in radians.
    """
    if len(points) < 3:
        return np.pi / 2
    
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi / 2

    max_radius = 0.0
    for face in hull.simplices:
        p0, p1, p2 = points[face[0]], points[face[1]], points[face[2]]
        
        # Check if the triangle is valid (non-degenerate)
        # Cross product magnitude squared
        c = np.cross(p1 - p0, p2 - p0)
        area_sq = np.dot(c, c)
        
        if area_sq < 1e-12:
            # Degenerate face, use centroid fallback
            centroid = (p0 + p1 + p2) / 3.0
            norm_c = np.linalg.norm(centroid)
            if norm_c < 1e-12:
                r_face = np.pi / 2
            else:
                centroid = centroid / norm_c
                d0 = np.arccos(np.clip(np.dot(centroid, p0), -1, 1))
                d1 = np.arccos(np.clip(np.dot(centroid, p1), -1, 1))
                d2 = np.arccos(np.clip(np.dot(centroid, p2), -1, 1))
                r_face = max(d0, d1, d2)
        else:
            # Circumcenter of the spherical triangle
            # The circumcenter C is the normalization of the 3D circumcenter of the chordal triangle.
            # Let O be the 3D circumcenter. O lies in the plane of the triangle.
            # O = p0 + u*(p1-p0) + v*(p2-p0)
            # |O-p0|^2 = |O-p1|^2 => 2*O.(p1-p0) = |p1|^2 - |p0|^2 + 2*p0.(p1-p0) ? No.
            # |O-p0|^2 = |O-p1|^2
            # O.O - 2 O.p0 + p0.p0 = O.O - 2 O.p1 + p1.p1
            # Since points are on unit sphere, p0.p0 = p1.p0 = 1.
            # -2 O.p0 + 1 = -2 O.p1 + 1 => O.p0 = O.p1
            # Similarly O.p0 = O.p2
            
            # Let O = a*p0 + b*p1 + c*p2 ? No, O is in the affine plane.
            # Let's use the formula for circumcenter of triangle with vertices on unit sphere.
            # The vector C (circumcenter direction) is proportional to:
            # C ∝ (p1 x p2) x (p0 x p2) ? No.
            
            # Standard formula:
            # Let n = (p1 - p0) x (p2 - p0)
            # The circumcenter O satisfies O . n = h (height of plane from origin)
            # And O is in the plane.
            
            # A robust way:
            # Solve for O in the basis of the plane.
            # Let e1 = p1 - p0, e2 = p2 - p0.
            # O = p0 + u e1 + v e2.
            # (O - p0)^2 = (O - p1)^2
            # (u e1 + v e2)^2 = (u e1 + v e2 - e1)^2
            # |u e1 + v e2|^2 = |u e1 + v e2|^2 - 2 (u e1 + v e2) . e1 + |e1|^2
            # 2 (u e1 + v e2) . e1 = |e1|^2
            # 2 u (e1.e1) + 2 v (e2.e1) = e1.e1
            
            # Similarly for e2:
            # 2 u (e1.e2) + 2 v (e2.e2) = e2.e2
            
            e1 = p1 - p0
            e2 = p2 - p0
            e1e1 = np.dot(e1, e1)
            e1e2 = np.dot(e1, e2)
            e2e2 = np.dot(e2, e2)
            
            det = e1e1 * e2e2 - e1e2 * e1e2
            if abs(det) < 1e-12:
                # Fallback
                centroid = (p0 + p1 + p2) / 3.0
                norm_c = np.linalg.norm(centroid)
                if norm_c < 1e-12:
                    r_face = np.pi / 2
                else:
                    centroid = centroid / norm_c
                    d0 = np.arccos(np.clip(np.dot(centroid, p0), -1, 1))
                    d1 = np.arccos(np.clip(np.dot(centroid, p1), -1, 1))
                    d2 = np.arccos(np.clip(np.dot(centroid, p2), -1, 1))
                    r_face = max(d0, d1, d2)
            else:
                u = (e1e1 * e2e2 - e1e2 * e2e2) / (2.0 * det)
                v = (e1e1 * e1e2 - e1e2 * e2e2) / (2.0 * det)
                
                O = p0 + u * e1 + v * e2
                norm_O = np.linalg.norm(O)
                if norm_O < 1e-12:
                    centroid = (p0 + p1 + p2) / 3.0
                    norm_c = np.linalg.norm(centroid)
                    if norm_c < 1e-12:
                        r_face = np.pi / 2
                    else:
                        centroid = centroid / norm_c
                        d0 = np.arccos(np.clip(np.dot(centroid, p0), -1, 1))
                        d1 = np.arccos(np.clip(np.dot(centroid, p1), -1, 1))
                        d2 = np.arccos(np.clip(np.dot(centroid, p2), -1, 1))
                        r_face = max(d0, d1, d2)
                else:
                    C = O / norm_O
                    d0 = np.arccos(np.clip(np.dot(C, p0), -1, 1))
                    d1 = np.arccos(np.clip(np.dot(C, p1), -1, 1))
                    d2 = np.arccos(np.clip(np.dot(C, p2), -1, 1))
                    r_face = max(d0, d1, d2)
        
        if r_face > max_radius:
            max_radius = r_face
            
    return max_radius

def _repulsion_grad(points, power=6):
    """
    Gradient of the sum of 1/d^power repulsion potential.
    """
    N_pts = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    dist_sq[dist_sq < 1e-12] = 1e-12
    
    mask = ~np.eye(N_pts, dtype=bool)
    
    # V = sum_{i<j} 1/d^power = 0.5 sum_{i!=j} 1/d^power
    # dV/dp_i = -power * sum_{j!=i} (p_i - p_j) / d_ij^(power+2)
    
    coeff = power * (dist_sq ** (-(power + 2) / 2.0))
    coeff[~mask] = 0.0
    
    # grad_i = - sum_j coeff[i,j] * (p_i - p_j)
    #         = - p_i * sum_j coeff[i,j] + sum_j coeff[i,j] * p_j
    
    sum_coeff = np.sum(coeff, axis=1)
    term1 = sum_coeff[:, np.newaxis] * points
    term2 = coeff @ points
    
    grad = -term1 + term2
    
    # Project onto tangent space
    dot_prod = np.sum(grad * points, axis=1, keepdims=True)
    grad_tan = grad - dot_prod * points
    
    return grad_tan

def _optimize_from_start(points, deadline):
    """
    Optimize a starting configuration using gradient descent with adaptive step size.
    """
    points = points.copy()
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
    
    lr = 0.01
    min_lr = 1e-8
    decay = 0.999
    
    # Use power=6 for steeper repulsion to avoid clustering
    for step in range(5000):
        if time.time() > deadline:
            break
            
        grad = _repulsion_grad(points, power=6)
        
        # Adaptive step size: scale by norm of gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
            
        # Normalize gradient direction
        grad_dir = grad / grad_norm
        
        # Update
        points_new = points + lr * grad_dir
        
        # Normalize
        norms = np.linalg.norm(points_new, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        points_new = points_new / norms
        
        # Check for improvement in a local metric? Or just trust the descent?
        # For simplicity, just update.
        points = points_new
        lr *= decay
        
        if lr < min_lr:
            break
            
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_points = None
    best_radius = np.pi / 2.0
    
    # 1. Fibonacci spiral initialization
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    inits = [fib_points]
    
    # 2. Random initializations
    rng = np.random.default_rng(42)
    for _ in range(5):
        vecs = rng.normal(size=(N, 3))
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        inits.append(vecs)
        
    for init_pts in inits:
        if time.time() > deadline - 10.0:
            break
            
        # Optimize
        points = _optimize_from_start(init_pts, deadline - 5.0)
        
        # Evaluate covering radius
        radius = _covering_radius(points)
        
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
            
    # If we have time, do a polish pass on the best points
    if best_points is not None and time.time() < deadline - 20.0:
        # Polish with smaller steps and different power
        polish_deadline = deadline - 5.0
        points = best_points.copy()
        
        # Use a finer optimization
        lr = 0.001
        for step in range(2000):
            if time.time() > polish_deadline:
                break
            grad = _repulsion_grad(points, power=8)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-14:
                break
            grad_dir = grad / grad_norm
            points = points + lr * grad_dir
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            norms[norms < 1e-15] = 1.0
            points = points / norms
            lr *= 0.995
            
        radius = _covering_radius(points)
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()

    if best_points is None:
        return fib_points
        
    return best_points

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

def _covering_radius(points):
    """
    Computes the covering radius in radians.
    """
    if len(points) < 3:
        return np.pi / 2
    
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi / 2

    max_radius = 0.0
    for face in hull.simplices:
        p0, p1, p2 = points[face[0]], points[face[1]], points[face[2]]
        
        e1 = p1 - p0
        e2 = p2 - p0
        e1e1 = np.dot(e1, e1)
        e1e2 = np.dot(e1, e2)
        e2e2 = np.dot(e2, e2)
        
        det = e1e1 * e2e2 - e1e2 * e1e2
        if abs(det) < 1e-12:
            centroid = (p0 + p1 + p2) / 3.0
            norm_c = np.linalg.norm(centroid)
            if norm_c < 1e-12:
                r_face = np.pi / 2
            else:
                centroid = centroid / norm_c
                d0 = np.arccos(np.clip(np.dot(centroid, p0), -1, 1))
                d1 = np.arccos(np.clip(np.dot(centroid, p1), -1, 1))
                d2 = np.arccos(np.clip(np.dot(centroid, p2), -1, 1))
                r_face = max(d0, d1, d2)
        else:
            u = (e1e1 * e2e2 - e1e2 * e2e2) / (2.0 * det)
            v = (e1e1 * e1e2 - e1e2 * e2e2) / (2.0 * det)
            O = p0 + u * e1 + v * e2
            norm_O = np.linalg.norm(O)
            if norm_O < 1e-12:
                centroid = (p0 + p1 + p2) / 3.0
                norm_c = np.linalg.norm(centroid)
                if norm_c < 1e-12:
                    r_face = np.pi / 2
                else:
                    centroid = centroid / norm_c
                    d0 = np.arccos(np.clip(np.dot(centroid, p0), -1, 1))
                    d1 = np.arccos(np.clip(np.dot(centroid, p1), -1, 1))
                    d2 = np.arccos(np.clip(np.dot(centroid, p2), -1, 1))
                    r_face = max(d0, d1, d2)
            else:
                C = O / norm_O
                d0 = np.arccos(np.clip(np.dot(C, p0), -1, 1))
                d1 = np.arccos(np.clip(np.dot(C, p1), -1, 1))
                d2 = np.arccos(np.clip(np.dot(C, p2), -1, 1))
                r_face = max(d0, d1, d2)
        
        if r_face > max_radius:
            max_radius = r_face
            
    return max_radius

def _repulsion_grad(points, power=6):
    N_pts = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    dist_sq[dist_sq < 1e-12] = 1e-12
    
    mask = ~np.eye(N_pts, dtype=bool)
    
    coeff = power * (dist_sq ** (-(power + 2) / 2.0))
    coeff[~mask] = 0.0
    
    sum_coeff = np.sum(coeff, axis=1)
    term1 = sum_coeff[:, np.newaxis] * points
    term2 = coeff @ points
    
    grad = -term1 + term2
    
    dot_prod = np.sum(grad * points, axis=1, keepdims=True)
    grad_tan = grad - dot_prod * points
    
    return grad_tan

def _optimize_from_start(points, deadline):
    points = points.copy()
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
    
    lr = 0.01
    min_lr = 1e-8
    decay = 0.999
    
    for step in range(5000):
        if time.time() > deadline:
            break
            
        grad = _repulsion_grad(points, power=6)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
            
        grad_dir = grad / grad_norm
        points_new = points + lr * grad_dir
        
        norms = np.linalg.norm(points_new, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        points_new = points_new / norms
        
        points = points_new
        lr *= decay
        
        if lr < min_lr:
            break
            
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_points = None
    best_radius = np.pi / 2.0
    
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    inits = [fib_points]
    
    rng = np.random.default_rng(42)
    for _ in range(5):
        vecs = rng.normal(size=(N, 3))
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        inits.append(vecs)
        
    for init_pts in inits:
        if time.time() > deadline - 10.0:
            break
            
        points = _optimize_from_start(init_pts, deadline - 5.0)
        radius = _covering_radius(points)
        
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
            
    if best_points is not None and time.time() < deadline - 20.0:
        polish_deadline = deadline - 5.0
        points = best_points.copy()
        
        lr = 0.001
        for step in range(2000):
            if time.time() > polish_deadline:
                break
            grad = _repulsion_grad(points, power=8)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-14:
                break
            grad_dir = grad / grad_norm
            points = points + lr * grad_dir
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            norms[norms < 1e-15] = 1.0
            points = points / norms
            lr *= 0.995
            
        radius = _covering_radius(points)
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()

    if best_points is None:
        return fib_points
        
    return best_points
# EVOLVE-BLOCK-END
