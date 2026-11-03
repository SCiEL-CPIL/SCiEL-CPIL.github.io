import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def _facet_circumradii(pts, simplices):
    """Vectorized computation of circumradii for all facets.
    Returns array of radii (radians) and array of circumcenter unit vectors."""
    a = pts[simplices[:, 0]]
    b = pts[simplices[:, 1]]
    c = pts[simplices[:, 2]]
    n = np.cross(a - b, a - c)
    n_norm = np.linalg.norm(n, axis=1)
    mask = n_norm > 1e-10
    safe_norm = np.maximum(n_norm, 1e-10)
    n = n / safe_norm[:, None]
    centroids = (a + b + c) / 3.0
    signs = np.where(np.einsum('ij,ij->i', n, centroids) < 0, -1.0, 1.0)
    n = n * signs[:, None]
    dots = np.clip(np.einsum('ij,ij->i', n, a), -1.0, 1.0)
    radii = np.arccos(dots)
    radii[~mask] = 0.0
    return radii, n, mask

def _covering_radius_deg(pts):
    """Exact covering radius in degrees via convex hull facet circumradii."""
    try:
        hull = ConvexHull(pts)
        radii, _, _ = _facet_circumradii(pts, hull.simplices)
        return float(np.degrees(np.max(radii)))
    except Exception:
        return 180.0

def _covering_radius_rad(pts):
    try:
        hull = ConvexHull(pts)
        radii, _, _ = _facet_circumradii(pts, hull.simplices)
        return float(np.max(radii))
    except Exception:
        return np.pi

def _fibonacci_spiral(offset=0.0):
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + offset
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)

def _electrostatic_refine(pts, n_iters=500, lr=0.01):
    """Repulsion-based refinement: each point is pushed away by all others via Coulomb-like force."""
    pts = pts.copy()
    for it in range(n_iters):
        D = pts @ pts.T
        chord2 = 2.0 - 2.0 * D
        np.fill_diagonal(chord2, 1.0)
        W = 1.0 / chord2
        np.fill_diagonal(W, 0.0)
        row_sum = W.sum(axis=1)
        forces = pts * row_sum[:, None] - (W @ pts)
        forces = forces - np.einsum('ij,ij->i', forces, pts)[:, None] * pts
        norms = np.linalg.norm(forces, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        forces = forces / norms
        pts = pts + lr * forces
        pts = _normalize(pts)
    return pts

def _smooth_surrogate(pts, beta=8.0):
    """Smooth surrogate: log-sum-exp of squared facet radii (soft-max).
    Returns (surrogate_value, weights, radii, circumcenters, simplices) or None."""
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None
    radii, cc, mask = _facet_circumradii(pts, hull.simplices)
    if not np.any(mask):
        return None
    valid = mask
    r_valid = radii[valid]
    cc_valid = cc[valid]
    simp_valid = hull.simplices[valid]
    r2 = r_valid * r_valid
    m = np.max(r2)
    exps = np.exp((r2 - m) * beta)
    s = np.sum(exps)
    if s < 1e-300:
        weights = np.ones_like(exps) / len(exps)
    else:
        weights = exps / s
    surrogate = m + np.log(s) / beta
    return surrogate, weights, r_valid, cc_valid, simp_valid

def _surrogate_gradient(pts, weights, simplices, circumcenters, beta=8.0):
    """Gradient of smooth surrogate w.r.t. each point (in R^3, then projected to tangent)."""
    grad = np.zeros_like(pts)
    for k in range(len(simplices)):
        w = weights[k]
        if w < 1e-8:
            continue
        c = circumcenters[k]
        a = pts[simplices[k, 0]]
        cos_r = np.clip(np.dot(c, a), -1.0, 1.0)
        sin_r = np.sqrt(max(1.0 - cos_r * cos_r, 1e-12))
        r_k = np.arccos(cos_r)
        for j in range(3):
            idx = simplices[k, j]
            v = pts[idx]
            d = c - v
            d_t = d - np.dot(d, v) * v
            d_t_norm = np.linalg.norm(d_t)
            if d_t_norm < 1e-10:
                continue
            d_t = d_t / d_t_norm
            mag = 2.0 * r_k / sin_r * w
            grad[idx] -= mag * d_t
    grad = grad - np.einsum('ij,ij->i', grad, pts)[:, None] * pts
    return grad

def _lloyd_iteration(points):
    """
    Perform one iteration of Lloyd's algorithm.
    1. Compute convex hull.
    2. For each point, find the circumcenters of the triangles incident to it.
    3. Move the point to the normalized sum of these circumcenters.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        
        sums = np.zeros((N, 3))
        counts = np.zeros(N)
        
        for tri in facets:
            i, j, k = tri
            p1 = points[i]
            p2 = points[j]
            p3 = points[k]
            
            v1 = p2 - p1
            v2 = p3 - p1
            cross = np.cross(v1, v2)
            norm_cross = np.linalg.norm(cross)
            
            if norm_cross < 1e-12:
                continue
                
            c = cross / norm_cross
            
            if np.dot(c, p1) < 0:
                c = -c
            
            sums[i] += c
            sums[j] += c
            sums[k] += c
            
            counts[i] += 1
            counts[j] += 1
            counts[k] += 1
        
        new_points = points.copy()
        for i in range(N):
            if counts[i] > 0:
                new_points[i] = sums[i]
                norm = np.linalg.norm(new_points[i])
                if norm > 1e-12:
                    new_points[i] /= norm
                else:
                    new_points[i] = points[i]
            
        return new_points
    except Exception:
        return points

def _gradient_descent(points, start_time, deadline, initial_lr=0.005, min_lr=1e-6, initial_temperature=0.05):
    """
    Optimize using gradient descent on a smooth surrogate objective.
    Uses log-sum-exp for soft-max of circumradii with adaptive temperature.
    """
    current_points = points.copy()
    best_points = current_points.copy()
    best_radius = _covering_radius_deg(current_points)
    lr = initial_lr
    temperature = initial_temperature
    stall_count = 0
    max_stall = 30
    
    while time.time() < deadline:
        surr = _smooth_surrogate(current_points, beta=1.0/temperature)
        if surr is None:
            break
        obj_val, weights, r_valid, cc_valid, simp_valid = surr
        
        grad = _surrogate_gradient(current_points, weights, simp_valid, cc_valid, beta=1.0/temperature)
        
        new_points = current_points - lr * grad
        new_points = _normalize(new_points)
        
        current_radius = _covering_radius_deg(new_points)
        
        if current_radius < best_radius - 1e-12:
            best_radius = current_radius
            best_points = new_points.copy()
            current_points = new_points.copy()
            stall_count = 0
            lr *= 0.998
            temperature *= 0.995
            if temperature < 0.005:
                temperature = 0.005
        else:
            stall_count += 1
            lr *= 0.92
            if stall_count > max_stall:
                hull = ConvexHull(current_points)
                facets = hull.simplices
                worst_idx = 0
                worst_r = 0
                for idx, tri in enumerate(facets):
                    p1, p2, p3 = current_points[tri[0]], current_points[tri[1]], current_points[tri[2]]
                    v1, v2 = p2 - p1, p3 - p1
                    cross = np.cross(v1, v2)
                    nc = np.linalg.norm(cross)
                    if nc < 1e-12:
                        continue
                    c = cross / nc
                    if np.dot(c, p1) < 0:
                        c = -c
                    dot_val = np.clip(np.dot(c, p1), -1.0, 1.0)
                    r = np.arccos(dot_val)
                    if r > worst_r:
                        worst_r = r
                        worst_idx = idx
                tri = facets[worst_idx]
                for vi in tri:
                    noise = np.random.normal(0, 0.01, 3)
                    current_points[vi] += noise
                current_points = _normalize(current_points)
                stall_count = 0
                lr = max(lr, 0.001)
            if lr < min_lr:
                break
        
        if time.time() > deadline - 5:
            break
    
    return best_points, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 142.0

    # Start with Fibonacci spiral
    points = _fibonacci_spiral()
    points = _normalize(points)
    
    best_points = points.copy()
    best_radius = _covering_radius_deg(points)
    
    # Phase 1: Run Lloyd's algorithm for initial convergence
    lloyd_deadline = start_time + 60.0
    stall_count = 0
    max_stall = 15
    
    while time.time() < lloyd_deadline:
        new_points = _lloyd_iteration(points)
        new_points = _normalize(new_points)
        
        current_radius = _covering_radius_deg(new_points)
        
        if current_radius < best_radius - 1e-10:
            best_radius = current_radius
            best_points = new_points.copy()
            points = new_points.copy()
            stall_count = 0
        else:
            points = new_points.copy()
            stall_count += 1
            
            if stall_count > max_stall:
                num_perturb = max(5, N // 10)
                indices = np.random.choice(N, num_perturb, replace=False)
                noise = np.random.normal(0, 0.02, (num_perturb, 3))
                points[indices] += noise
                points[indices] = _normalize(points[indices])
                stall_count = 0
    
    # Phase 2: Gradient descent with smooth surrogate
    grad_deadline = start_time + 118.0
    points = best_points.copy()
    gd_points, gd_radius = _gradient_descent(points, start_time, grad_deadline, 
                                              initial_lr=0.003, min_lr=1e-7, initial_temperature=0.04)
    
    if gd_radius < best_radius:
        best_radius = gd_radius
        best_points = gd_points.copy()
    
    # Phase 3: Final high-precision polish with Lloyd's + targeted perturbation
    polish_deadline = start_time + 142.0
    points = best_points.copy()
    stall_count = 0
    max_stall = 20
    
    while time.time() < polish_deadline:
        new_points = _lloyd_iteration(points)
        new_points = _normalize(new_points)
        
        current_radius = _covering_radius_deg(new_points)
        
        if current_radius < best_radius - 1e-12:
            best_radius = current_radius
            best_points = new_points.copy()
            points = new_points.copy()
            stall_count = 0
        else:
            stall_count += 1
            if stall_count > max_stall:
                try:
                    hull = ConvexHull(points)
                    facets = hull.simplices
                    worst_idx = 0
                    worst_r = 0
                    for idx, tri in enumerate(facets):
                        p1, p2, p3 = points[tri[0]], points[tri[1]], points[tri[2]]
                        v1, v2 = p2 - p1, p3 - p1
                        cross = np.cross(v1, v2)
                        nc = np.linalg.norm(cross)
                        if nc < 1e-12:
                            continue
                        c = cross / nc
                        if np.dot(c, p1) < 0:
                            c = -c
                        dot_val = np.clip(np.dot(c, p1), -1.0, 1.0)
                        r = np.arccos(dot_val)
                        if r > worst_r:
                            worst_r = r
                            worst_idx = idx
                    tri = facets[worst_idx]
                    for vi in tri:
                        noise = np.random.normal(0, 0.005, 3)
                        points[vi] += noise
                    points = _normalize(points)
                except Exception:
                    pass
                stall_count = 0
                if time.time() > polish_deadline - 3:
                    break
    
    return best_points
