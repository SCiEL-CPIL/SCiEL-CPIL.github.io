import numpy as np
import scipy.spatial
import time
import math

N = 50

def _compute_facet_data(points):
    """
    Computes the circumradius and circumcenter for each facet of the convex hull.
    Returns:
        radii: array of shape (n_facets,)
        centers: array of shape (n_facets, 3)
        cos_radii: array of shape (n_facets,)
    """
    # Ensure points are on the sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    try:
        hull = scipy.spatial.ConvexHull(points)
    except Exception:
        # Fallback: return dummy data
        return np.array([np.pi]), np.array([[0.0, 0.0, 0.0]]), np.array([1.0])

    n_facets = len(hull.simplices)
    radii = np.zeros(n_facets)
    centers = np.zeros((n_facets, 3))
    cos_radii = np.zeros(n_facets)
    
    for i, simplex in enumerate(hull.simplices):
        a = points[simplex[0]]
        b = points[simplex[1]]
        c = points[simplex[2]]
        
        # Circumcenter of the spherical triangle is the normalized sum of vertices
        # This is valid if the origin is inside the tetrahedron (0, a, b, c)
        c_dir = a + b + c
        norm_c_dir = np.linalg.norm(c_dir)
        
        if norm_c_dir < 1e-10:
            # Degenerate case, fallback to centroid
            c_center = (a + b + c) / 3.0
            norm_c = np.linalg.norm(c_center)
            if norm_c < 1e-10:
                radii[i] = 0.0
                centers[i] = 0.0
                cos_radii[i] = 1.0
                continue
            c_center = c_center / norm_c
        else:
            c_center = c_dir / norm_c_dir
            
        centers[i] = c_center
        
        # Radius is arccos(dot(c_center, a))
        cos_r = np.dot(c_center, a)
        cos_r = np.clip(cos_r, -1.0, 1.0)
        cos_radii[i] = cos_r
        radii[i] = np.arccos(cos_r)
        
    return radii, centers, cos_radii

def _soft_max_radius_and_grad(points, tau=0.05):
    """
    Computes the soft-max covering radius and gradients.
    """
    radii, centers, cos_radii = _compute_facet_data(points)
    n_facets = len(radii)
    
    if n_facets == 0:
        return np.pi, np.zeros_like(points)
        
    max_r = np.max(radii)
    if max_r < 1e-10:
        return 0.0, np.zeros_like(points)
    
    # Soft max
    exp_terms = np.exp((radii - max_r) / tau)
    sum_exp = np.sum(exp_terms)
    soft_max_r = max_r + tau * np.log(sum_exp)
    
    # Weights
    weights = exp_terms / sum_exp
    
    # Gradients
    grads = np.zeros_like(points)
    
    for i in range(n_facets):
        w = weights[i]
        if w < 1e-12:
            continue
        r = radii[i]
        if r < 1e-10:
            continue
        sin_r = np.sin(r)
        if sin_r < 1e-10:
            continue
            
        C = centers[i]
        cos_r = cos_radii[i]
        
        simplex = scipy.spatial.ConvexHull(points).simplices[i] # This is inefficient, reusing hull is better
        # Note: We should pass the hull or precompute simplex indices to avoid recomputing hull
        # For now, let's assume we have the hull from _compute_facet_data. 
        # I will refactor _compute_facet_data to return hull too or just indices.
        
        # Actually, let's modify _compute_facet_data to return the hull as well or just the simplices.
        # Let's create a helper that returns everything.
        
    # Refactor: We need the simplices. Let's compute them once.
    # I'll rewrite the logic to be more efficient.
    
    return soft_max_r, grads

def _compute_soft_max_and_grads(points, tau=0.05):
    """
    Efficiently computes soft-max radius and gradients.
    """
    # Ensure points are on the sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    try:
        hull = scipy.spatial.ConvexHull(points)
    except Exception:
        return np.pi, np.zeros_like(points)

    n_facets = len(hull.simplices)
    if n_facets == 0:
        return np.pi, np.zeros_like(points)
        
    radii = np.zeros(n_facets)
    centers = np.zeros((n_facets, 3))
    cos_radii = np.zeros(n_facets)
    
    # Precompute facet data
    for i, simplex in enumerate(hull.simplices):
        a = points[simplex[0]]
        b = points[simplex[1]]
        c = points[simplex[2]]
        
        c_dir = a + b + c
        norm_c_dir = np.linalg.norm(c_dir)
        
        if norm_c_dir < 1e-10:
            c_center = (a + b + c) / 3.0
            norm_c = np.linalg.norm(c_center)
            if norm_c < 1e-10:
                radii[i] = 0.0
                centers[i] = 0.0
                cos_radii[i] = 1.0
                continue
            c_center = c_center / norm_c
        else:
            c_center = c_dir / norm_c_dir
            
        centers[i] = c_center
        cos_r = np.dot(c_center, a)
        cos_r = np.clip(cos_r, -1.0, 1.0)
        cos_radii[i] = cos_r
        radii[i] = np.arccos(cos_r)
        
    max_r = np.max(radii)
    if max_r < 1e-10:
        return 0.0, np.zeros_like(points)
        
    exp_terms = np.exp((radii - max_r) / tau)
    sum_exp = np.sum(exp_terms)
    soft_max_r = max_r + tau * np.log(sum_exp)
    
    weights = exp_terms / sum_exp
    
    grads = np.zeros_like(points)
    
    for i in range(n_facets):
        w = weights[i]
        if w < 1e-12:
            continue
        r = radii[i]
        if r < 1e-10:
            continue
        sin_r = np.sin(r)
        if sin_r < 1e-10:
            continue
            
        C = centers[i]
        cos_r = cos_radii[i]
        
        simplex = hull.simplices[i]
        for k in range(3):
            p_idx = simplex[k]
            v = points[p_idx]
            # Gradient direction: (C - cos_r * v) / sin_r
            # This points towards C (reducing radius)
            # We want to minimize, so we move in this direction.
            # In gradient descent: p_new = p - step * grad
            # So grad should be the direction of INCREASE.
            # Direction of increase is away from C: -(C - cos_r * v) / sin_r
            # So grad = - (C - cos_r * v) / sin_r
            # Let's verify.
            # If we move v towards C, radius decreases.
            # So -grad should point towards C.
            # So grad should point away from C.
            # grad = (cos_r * v - C) / sin_r
            
            grads[p_idx] += w * (cos_r * v - C) / sin_r
            
    return soft_max_r, grads

def _adam_optimize(points, time_budget):
    """
    Optimizes points on the sphere using Adam optimizer.
    """
    current_points = points.copy()
    best_radius, _ = _compute_soft_max_and_grads(current_points, tau=0.05)
    best_points = current_points.copy()
    
    # Adam parameters
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    lr = 0.01
    m = np.zeros_like(current_points)
    v = np.zeros_like(current_points)
    t = 0
    
    start_time = time.time()
    
    while time.time() - start_time < time_budget:
        t += 1
        # Decay learning rate
        current_lr = lr / (1.0 + t * 0.001)
        
        radius, grads = _compute_soft_max_and_grads(current_points, tau=0.05)
        
        if radius < best_radius:
            best_radius = radius
            best_points = current_points.copy()
            
        # Update Adam moments
        m = beta1 * m + (1 - beta1) * grads
        v = beta2 * v + (1 - beta2) * (grads ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        # Update
        update = m_hat / (np.sqrt(v_hat) + epsilon)
        new_points = current_points - current_lr * update
        
        # Project to sphere
        new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
        
        current_points = new_points
        
    return best_points, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    best_points = None
    best_radius = np.inf
    
    seeds = []
    
    # 1. Fibonacci Sphere
    j = np.arange(N) + 0.5
    z = 1.0 - 2.0 * j / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * j
    r = np.sqrt(1.0 - z * z)
    fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    seeds.append(fib)
    
    # 2. Random starts
    rng = np.random.default_rng(42)
    for _ in range(5):
        pts = rng.normal(size=(N, 3))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        seeds.append(pts)
        
    # Process seeds
    for i, seed_pts in enumerate(seeds):
        if time.time() > deadline - 5:
            break
            
        budget = min(10.0, deadline - 5 - time.time())
        if budget <= 0:
            break
            
        pts, rad = _adam_optimize(seed_pts, time_budget=budget)
        
        if rad < best_radius:
            best_radius = rad
            best_points = pts.copy()
            
    # If we have time left, refine the best point found
    if best_points is not None and time.time() < deadline - 5:
        budget = deadline - 5 - time.time()
        pts, rad = _adam_optimize(best_points, time_budget=budget)
        if rad < best_radius:
            best_radius = rad
            best_points = pts

    if best_points is None:
        best_points = seeds[0]
        
    return best_points
