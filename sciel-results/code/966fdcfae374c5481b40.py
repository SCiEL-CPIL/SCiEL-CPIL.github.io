import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def compute_covering_radius_and_facets(points):
    """
    Computes the covering radius and returns the radii and indices for each facet.
    points: (N, 3) array of unit vectors.
    """
    try:
        hull = ConvexHull(points)
        simplices = hull.simplices
    except Exception:
        return np.pi, np.array([]), np.array([]), np.array([])
    
    if len(simplices) == 0:
        return np.pi, np.array([]), np.array([]), np.array([])
        
    # Extract vertices for all facets
    a = points[simplices[:, 0]]
    b = points[simplices[:, 1]]
    c = points[simplices[:, 2]]
    
    # Compute normals
    n = np.cross(b - a, c - a)
    norms = np.linalg.norm(n, axis=1)
    
    # Avoid division by zero
    valid = norms > 1e-10
    if not np.all(valid):
        norms[~valid] = 1.0
        n[~valid] = [0, 0, 1]
        
    u = n / norms[:, np.newaxis]
    
    # Orient normals to point outwards (away from origin)
    centroids = (a + b + c) / 3.0
    dots = np.sum(u * centroids, axis=1)
    u[dots < 0] *= -1
    
    # Circumradius is angle between u and a
    cos_theta = np.sum(u * a, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    radii = np.arccos(cos_theta)
    
    max_radius = np.max(radii)
    return max_radius, radii, simplices, u

def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return points / norms

def _soft_max_weights(radii, temperature=0.03):
    """
    Compute smooth weights based on radii using soft-max.
    Larger radii get higher weights, but all facets contribute smoothly.
    """
    shifted = radii - np.max(radii)
    exp_vals = np.exp(shifted / temperature)
    weights = exp_vals / np.sum(exp_vals)
    return weights

def _polish_local(points, max_iter=300, tol=1e-10):
    """
    Local polishing: optimize the worst facets by moving their vertices.
    Uses gradient descent on the smooth surrogate with adaptive step size.
    """
    best_points = points.copy()
    best_radius = np.pi
    
    for it in range(max_iter):
        max_radius, radii, simplices, centers = compute_covering_radius_and_facets(points)
        
        if len(simplices) == 0:
            break
        
        if max_radius < best_radius:
            best_radius = max_radius
            best_points = points.copy()
        
        # Focus on the top-k largest facets
        k = min(10, len(radii))
        top_k_idx = np.argsort(radii)[-k:]
        
        # Compute gradient-like update for vertices of top-k facets
        updates = np.zeros_like(points)
        
        for idx in top_k_idx:
            simplex = simplices[idx]
            i_idx, j_idx, k_idx = simplex
            center = centers[idx]
            
            for v_idx in [i_idx, j_idx, k_idx]:
                p = points[v_idx]
                # Tangent component of direction to circumcenter
                tangent = center - np.dot(center, p) * p
                norm_t = np.linalg.norm(tangent)
                if norm_t > 1e-10:
                    tangent /= norm_t
                    updates[v_idx] += tangent
        
        # Normalize updates
        update_norms = np.linalg.norm(updates, axis=1, keepdims=True)
        update_norms = np.maximum(update_norms, 1e-10)
        updates = updates / update_norms
        
        # Adaptive step size with cooling
        step = 0.005 * (1.0 + 0.5 * np.cos(it * 0.1))
        
        new_points = points + step * updates
        new_points = _normalize(new_points)
        
        # Check if improvement
        new_max_radius, _, _, _ = compute_covering_radius_and_facets(new_points)
        if new_max_radius < max_radius:
            points = new_points
        else:
            # Reduce step and try again
            step *= 0.5
            new_points = points + step * updates
            new_points = _normalize(new_points)
            new_max_radius, _, _, _ = compute_covering_radius_and_facets(new_points)
            if new_max_radius < max_radius:
                points = new_points
            else:
                if step < 1e-8:
                    break
    
    return best_points, best_radius

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    best_points = None
    best_radius = np.pi
    
    # Initial configuration: Fibonacci Sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    init_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    points = init_fib.copy()
    
    # Phase 1: Global Lloyd-like iteration with smooth soft-max weights
    max_iter = 1500
    alpha_base = 0.05
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit - 25.0:
            break
        
        max_radius, radii, simplices, centers = compute_covering_radius_and_facets(points)
        
        if len(simplices) == 0:
            break
            
        if max_radius < best_radius:
            best_radius = max_radius
            best_points = points.copy()
        
        # Calculate update using soft-max weights for smooth contribution from all facets
        updates = np.zeros_like(points)
        
        weights = _soft_max_weights(radii, temperature=0.03)
        
        for m in range(len(simplices)):
            w_m = weights[m]
            if w_m < 1e-6:
                continue
                
            simplex = simplices[m]
            i_idx, j_idx, k_idx = simplex
            center = centers[m]
            
            for v_idx in [i_idx, j_idx, k_idx]:
                p = points[v_idx]
                # Use tangent component for better behavior on sphere
                tangent = center - np.dot(center, p) * p
                updates[v_idx] += w_m * tangent
        
        # Normalize updates
        update_norms = np.linalg.norm(updates, axis=1, keepdims=True)
        update_norms = np.maximum(update_norms, 1e-10)
        updates = updates / update_norms
        
        # Adaptive step size: decay over time
        progress = iteration / max_iter
        alpha = alpha_base * (1.0 - 0.7 * progress)
        
        points = points + alpha * updates
        points = _normalize(points)
        
        # Occasional random perturbation to escape local minima
        if iteration % 150 == 75 and iteration < max_iter - 300:
            if time.time() - start_time < time_limit - 35.0:
                perturbation = np.random.randn(N, 3) * 0.008
                perturbed = _normalize(points + perturbation)
                perturbed_radius, _, _, _ = compute_covering_radius_and_facets(perturbed)
                if perturbed_radius < best_radius * 0.998:
                    points = perturbed
    
    # Phase 2: Local polishing with tighter focus on worst facets
    if time.time() - start_time < time_limit - 5.0:
        polished_points, polished_radius = _polish_local(points, max_iter=300)
        if polished_radius < best_radius:
            best_radius = polished_radius
            best_points = polished_points
    
    if best_points is None:
        best_points = init_fib
    
    return best_points
