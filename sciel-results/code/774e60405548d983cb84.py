import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def compute_facet_data(points):
    """
    Computes the circumradius, circumcenter direction, and facet area for each facet.
    Returns:
        radii: array of shape (M,)
        centers: array of shape (M, 3) unit vectors
        simplices: array of shape (M, 3)
        areas: array of shape (M,)
    """
    try:
        hull = ConvexHull(points)
        simplices = hull.simplices
    except Exception:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    if len(simplices) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
        
    a = points[simplices[:, 0]]
    b = points[simplices[:, 1]]
    c = points[simplices[:, 2]]
    
    # Compute normals
    n = np.cross(b - a, c - a)
    norms = np.linalg.norm(n, axis=1)
    
    # Facet area = 0.5 * |n|
    areas = 0.5 * norms
    
    valid = norms > 1e-10
    
    M = len(simplices)
    radii = np.full(M, np.pi)
    centers = np.zeros((M, 3))
    centers[~valid] = [0, 0, 1]
    
    if not np.all(valid):
        if np.any(valid):
            a_v = a[valid]
            b_v = b[valid]
            c_v = c[valid]
            n_v = n[valid]
            norms_v = norms[valid]
            
            u = n_v / norms_v[:, np.newaxis]
            centroids = (a_v + b_v + c_v) / 3.0
            dots = np.sum(u * centroids, axis=1)
            u[dots < 0] *= -1
            
            cos_theta = np.sum(u * a_v, axis=1)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            radii[valid] = np.arccos(cos_theta)
            centers[valid] = u
            
        return radii, centers, simplices, areas

    u = n / norms[:, np.newaxis]
    
    centroids = (a + b + c) / 3.0
    dots = np.sum(u * centroids, axis=1)
    u[dots < 0] *= -1
    
    cos_theta = np.sum(u * a, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    radii = np.arccos(cos_theta)
    
    return radii, u, simplices, areas

def compute_voronoi_centroids(points):
    """
    Compute the area-weighted Voronoi centroid for each point using convex hull facets.
    Each point's Voronoi cell is approximated by the union of its adjacent facets'
    circumcenter contributions weighted by facet area.
    """
    radii, centers, simplices, areas = compute_facet_data(points)
    if len(radii) == 0:
        return points.copy()
    
    centroids = np.zeros_like(points)
    weights = np.zeros(N)
    
    for m in range(len(centers)):
        idx = simplices[m]
        i_idx, j_idx, k_idx = idx
        w = areas[m]
        centroids[i_idx] += w * centers[m]
        centroids[j_idx] += w * centers[m]
        centroids[k_idx] += w * centers[m]
        weights[i_idx] += w
        weights[j_idx] += w
        weights[k_idx] += w
    
    valid = weights > 1e-10
    centroids[valid] /= weights[valid, np.newaxis]
    
    # Normalize to unit sphere
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    centroids /= norms
    
    return centroids

def lloyd_step(points, alpha=0.5):
    """
    Spherical Lloyd's step: move each point toward its area-weighted Voronoi centroid.
    """
    centroids = compute_voronoi_centroids(points)
    
    new_points = points + alpha * (centroids - points)
    norms = np.linalg.norm(new_points, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    new_points /= norms
    
    return new_points

def smooth_objective_and_grad(points):
    """
    Compute a smooth surrogate for the max facet radius and its gradient.
    Uses soft-max of radii with proper gradient computation.
    Returns:
        value: float
        grad: array of shape (N, 3)
    """
    radii, centers, simplices, areas = compute_facet_data(points)
    if len(radii) == 0:
        return 0.0, np.zeros_like(points)
    
    beta = 50.0
    
    # Soft-max of radii
    max_r = np.max(radii)
    scaled = beta * (radii - max_r)
    exp_scaled = np.exp(scaled - np.max(scaled))
    sum_exp = np.sum(exp_scaled)
    weights = exp_scaled / sum_exp
    
    # Soft max value: max_r + log(sum_exp)/beta
    value = max_r + np.log(sum_exp) / beta
    
    # Gradient: sum_m w_m * dr_m/dp
    # r_m = arccos(u_m . a_m)
    # dr_m/dp_i for vertex i in facet m:
    # If we treat u_m as constant: dr_m = -(u_m . dp_i) / sin(r_m)
    # So dr_m/dp_i = -u_m / sin(r_m) for each vertex i in facet m
    
    grad_points = np.zeros_like(points)
    
    for m in range(len(radii)):
        if weights[m] < 1e-14:
            continue
            
        i_idx, j_idx, k_idx = simplices[m]
        center = centers[m]
        r = radii[m]
        
        sin_r = np.sin(r)
        if sin_r < 1e-8:
            continue
            
        coeff = -weights[m] / sin_r
        
        grad_points[i_idx] += coeff * center
        grad_points[j_idx] += coeff * center
        grad_points[k_idx] += coeff * center
        
    return value, grad_points

def gradient_descent_step(points, alpha, beta):
    """
    One gradient descent step on the smooth objective, projected to tangent space.
    """
    val, grad = smooth_objective_and_grad(points)
    
    # Project to tangent space
    grad_tangent = grad - np.sum(grad * points, axis=1, keepdims=True) * points
    
    new_points = points - alpha * grad_tangent
    norms = np.linalg.norm(new_points, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    new_points /= norms
    
    return new_points, val

def fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def greedy_init(n_points, seed=42):
    np.random.seed(seed)
    points = [np.random.randn(3)]
    points[0] /= np.linalg.norm(points[0])
    
    for _ in range(n_points - 1):
        candidates = np.random.randn(2000, 3)
        candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)
        
        existing = np.array(points)
        dots = candidates @ existing.T
        min_cos = np.min(dots, axis=1)
        min_cos = np.clip(min_cos, -1.0, 1.0)
        max_dist = np.arccos(min_cos)
        
        best_idx = np.argmax(max_dist)
        points.append(candidates[best_idx])
        
    return np.array(points)

def polish(points, time_budget):
    """
    High-precision polish: alternate Lloyd steps and gradient descent with adaptive step sizes.
    """
    start = time.time()
    best_local = points.copy()
    best_local_radius = np.pi
    
    # Compute initial radius
    radii, _, _, _ = compute_facet_data(points)
    if len(radii) > 0:
        best_local_radius = np.max(radii)
    
    alpha_lloyd = 0.3
    alpha_grad = 0.02
    
    for step in range(5000):
        if time.time() - start > time_budget:
            break
        
        # Alternate between Lloyd and gradient steps
        if step % 3 == 0:
            points = lloyd_step(points, alpha=alpha_lloyd)
        else:
            points, val = gradient_descent_step(points, alpha_grad, 50.0)
        
        # Check improvement
        radii, _, _, _ = compute_facet_data(points)
        if len(radii) > 0:
            max_r = np.max(radii)
            if max_r < best_local_radius:
                best_local_radius = max_r
                best_local = points.copy()
                # Decay step sizes
                alpha_lloyd *= 0.98
                alpha_grad *= 0.98
            else:
                alpha_lloyd *= 0.995
                alpha_grad *= 0.995
        
        if alpha_lloyd < 1e-6 and alpha_grad < 1e-6:
            break
    
    return best_local

def construct():
    start_time = time.time()
    time_limit = 145.0
    
    best_points = None
    best_radius = np.pi
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Fibonacci Sphere
    initializations.append(fibonacci_sphere(N))
    
    # 2. Greedy with different seeds
    initializations.append(greedy_init(N, seed=7))
    initializations.append(greedy_init(N, seed=123))
    
    # 3. Fibonacci + noise
    for noise_scale in [0.1, 0.2, 0.3]:
        np.random.seed(42 + int(noise_scale * 100))
        init = fibonacci_sphere(N) + np.random.randn(N, 3) * noise_scale
        init /= np.linalg.norm(init, axis=1, keepdims=True)
        initializations.append(init)
    
    # 4. Random
    np.random.seed(99)
    init_rand = np.random.randn(N, 3)
    init_rand /= np.linalg.norm(init_rand, axis=1, keepdims=True)
    initializations.append(init_rand)
    
    max_restarts = len(initializations)
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit - 10.0:
            break
            
        points = initializations[restart].copy()
        
        # Optimization: alternate Lloyd and gradient descent
        alpha_lloyd = 0.3
        alpha_grad = 0.02
        no_improve_count = 0
        
        while True:
            if time.time() - start_time > time_limit - 10.0:
                break
            
            # Compute current radius
            radii, _, _, _ = compute_facet_data(points)
            if len(radii) == 0:
                break
            
            max_radius = np.max(radii)
            
            if max_radius < best_radius:
                best_radius = max_radius
                best_points = points.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > 80:
                    break
                if no_improve_count > 40:
                    alpha_lloyd *= 0.95
                    alpha_grad *= 0.95
            
            # Alternate steps
            if no_improve_count % 3 == 0:
                points = lloyd_step(points, alpha=alpha_lloyd)
            else:
                points, _ = gradient_descent_step(points, alpha_grad, 50.0)
            
            # Normalize
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            points /= norms
            
            # Slow decay
            alpha_lloyd *= 0.998
            alpha_grad *= 0.998
            
            if alpha_lloyd < 1e-6 and alpha_grad < 1e-6:
                break
    
    # Final polish
    if best_points is not None:
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time > 3.0:
            polished = polish(best_points, remaining_time)
            radii, _, _, _ = compute_facet_data(polished)
            if len(radii) > 0:
                max_r = np.max(radii)
                if max_r < best_radius:
                    best_radius = max_r
                    best_points = polished
    else:
        best_points = initializations[0]
        
    return best_points
