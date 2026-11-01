import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50

def _normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return points / norms

def _fib_sample(n):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _compute_hull_data(p):
    """
    Compute convex hull facets and their circumcenters/radii (vectorized).
    Returns facets, circumcenters, radii.
    """
    try:
        hull = ConvexHull(p)
        facets = hull.simplices
    except Exception:
        return None, None, None
    
    n_facets = len(facets)
    if n_facets == 0:
        return None, None, None
    
    # Vectorized circumcenter computation
    a = p[facets[:, 0]]  # (F, 3)
    b = p[facets[:, 1]]
    c = p[facets[:, 2]]
    
    v1 = a - b  # (F, 3)
    v2 = b - c
    n = np.cross(v1, v2)  # (F, 3)
    norm_n = np.linalg.norm(n, axis=1)  # (F,)
    
    # Handle degenerate facets
    valid = norm_n > 1e-12
    circumcenters = np.zeros((n_facets, 3))
    radii = np.zeros(n_facets)
    
    if np.any(valid):
        cc = n[valid] / norm_n[valid, np.newaxis]
        # Ensure circumcenter faces outward (toward 'a')
        dots = np.sum(cc * a[valid], axis=1)
        cc[dots < 0] *= -1
        circumcenters[valid] = cc
        
        cos_r = np.clip(np.sum(cc * a[valid], axis=1), -1.0, 1.0)
        radii[valid] = np.arccos(cos_r)
    
    return facets, circumcenters, radii

def _covering_radius_deg_hull(config):
    """
    Compute the exact covering radius in degrees using convex hull facets.
    """
    facets, circumcenters, radii = _compute_hull_data(config)
    if radii is None:
        # Fallback to sampling if hull fails
        grid = _fib_sample(20000)
        dots = grid @ config.T
        min_dots = np.min(dots, axis=1)
        max_angle = np.arccos(np.clip(min_dots, -1.0, 1.0))
        return np.degrees(np.max(max_angle))
    
    return np.degrees(np.max(radii))

def _soft_max_objective(params, beta=100.0):
    """
    Soft-max surrogate: approximates max(radii) via log-sum-exp.
    As beta -> inf, this approaches the true max.
    """
    X = params.reshape(N, 3)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    X = X / norms
    
    facets, circumcenters, radii = _compute_hull_data(X)
    if facets is None:
        return 1e10
        
    r_max = np.max(radii)
    if r_max < 1e-15:
        return r_max
    shifted = beta * (radii - r_max)
    log_sum_exp = np.log(np.sum(np.exp(shifted))) / beta + r_max
    return log_sum_exp

def _repulsion_relax(points, alpha_init=0.05, decay=0.995, max_iter=300, deadline=None):
    """
    Repulsion Relaxation (Thomson Problem solver).
    Minimize sum 1/||p_i - p_j||^2 to spread points evenly.
    """
    points = points.copy()
    alpha = alpha_init
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        # Calculate forces
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        
        # Avoid division by zero
        mask = np.eye(N, dtype=bool)
        inv_dist_sq = np.zeros_like(dist_sq)
        np.divide(1.0, dist_sq, out=inv_dist_sq, where=~mask)
        
        # Force F_i = sum_j (p_i - p_j) / ||p_i - p_j||^2
        forces = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        
        # Project onto tangent plane: F_tan = F - (F . p) p
        dot_fp = np.sum(forces * points, axis=1)
        forces_tan = forces - dot_fp[:, np.newaxis] * points
        
        # Update
        points = points + alpha * forces_tan
        
        # Normalize
        norms = np.linalg.norm(points, axis=1)
        points = points / norms[:, np.newaxis]
        
        # Decay alpha
        alpha *= decay
        
    return points

def _lloyd_step_vectorized(config):
    """
    Perform one Lloyd iteration (vectorized).
    Moves each point towards the spherical mean of the circumcenters of the faces for which it is the nearest site.
    """
    try:
        hull = ConvexHull(config)
    except Exception:
        return config
    
    facets = hull.simplices
    a = config[facets[:, 0]]
    b = config[facets[:, 1]]
    c = config[facets[:, 2]]
    
    v1 = a - b
    v2 = b - c
    n = np.cross(v1, v2)
    norm_n = np.linalg.norm(n, axis=1)
    
    valid = norm_n > 1e-12
    if not np.any(valid):
        return config
    
    cc = n[valid] / norm_n[valid, np.newaxis]
    dots = np.sum(cc * a[valid], axis=1)
    cc[dots < 0] *= -1
    
    # For each circumcenter, find the nearest site
    dots_all = cc @ config.T
    nearest_idx = np.argmax(dots_all, axis=1)
    
    # Accumulate targets for each site
    sums = np.zeros((N, 3))
    counts = np.zeros(N)
    
    np.add.at(sums, nearest_idx, cc)
    np.add.at(counts, nearest_idx, 1)
    
    new_config = config.copy()
    for i in range(N):
        if counts[i] > 0:
            target = sums[i]
            n = np.linalg.norm(target)
            if n > 1e-10:
                new_config[i] = target / n
            else:
                new_config[i] = config[i]
        else:
            new_config[i] = config[i]
            
    return new_config

def _local_search(points, grid_points, deadline):
    """
    Local search: move points towards the worst point.
    """
    step_size = 0.1
    best_points = points.copy()
    best_radius = _covering_radius(points, grid_points)
    
    for iteration in range(1000):
        if time.time() > deadline - 5.0:
            break
        
        # Find the worst point
        dots = grid_points @ points.T
        dots = np.clip(dots, -1.0, 1.0)
        max_dots = np.max(dots, axis=1)
        worst_idx = np.argmin(max_dots)
        worst_point = grid_points[worst_idx]
        
        # Find the 3 nearest points to the worst point
        dots = points @ worst_point
        dots = np.clip(dots, -1.0, 1.0)
        nearest_indices = np.argsort(-dots)[:3]
        
        # Move these points towards the worst point
        for idx in nearest_indices:
            p = points[idx]
            direction = worst_point - p
            dot_dir_p = np.dot(direction, p)
            direction_tangent = direction - dot_dir_p * p
            norm_dir = np.linalg.norm(direction_tangent)
            if norm_dir > 1e-10:
                direction_tangent = direction_tangent / norm_dir
                points[idx] = p + step_size * direction_tangent
        
        points = _normalize(points)
        
        current_radius = _covering_radius(points, grid_points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
    
    return best_points

def _projected_gradient_polish(config, deadline, lr_init=0.01, lr_decay=0.98, max_iter=200):
    """
    Final polish using projected gradient descent on the tangent space.
    Directly minimizes the max angular distance.
    """
    config = config.copy()
    best_config = config.copy()
    best_score = _covering_radius_deg_hull(config)
    lr = lr_init
    
    for it in range(max_iter):
        if time.time() > deadline - 2.0:
            break
        
        # Compute gradient of max radius w.r.t. each point
        facets, circumcenters, radii = _compute_hull_data(config)
        if facets is None:
            break
        
        # Find which facet has the max radius
        max_idx = np.argmax(radii)
        max_facet = facets[max_idx]
        
        # The circumcenter of the max facet
        cc = circumcenters[max_idx]
        
        # For each vertex of the max facet, compute the gradient contribution
        # Moving a vertex p_i changes the circumcenter
        # Gradient of radius w.r.t. p_i: we want to move p_i to reduce the cap
        # The direction to move is away from the circumcenter
        grad = np.zeros((N, 3))
        
        for vi in max_facet:
            # The vertex vi should move in the direction that shrinks the facet
            # Move away from circumcenter, projected to tangent plane
            dir_vec = config[vi] - cc
            dir_vec = dir_vec / (np.linalg.norm(dir_vec) + 1e-15)
            # Project to tangent plane at config[vi]
            dot = np.dot(dir_vec, config[vi])
            grad[vi] = dir_vec - dot * config[vi]
        
        # Update
        config = config - lr * grad
        config = _normalize(config)
        
        current_score = _covering_radius_deg_hull(config)
        if current_score < best_score:
            best_score = current_score
            best_config = config.copy()
        
        lr *= lr_decay
    
    return best_config

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 120s limit, leave margin for overhead
    
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points = _normalize(points)
    
    # Create a grid of points on the sphere for evaluating the covering radius
    n_theta = 100
    n_phi = 200
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    theta_grid = theta_grid.flatten()
    phi_grid = phi_grid.flatten()
    
    grid_points = np.column_stack([
        np.sin(theta_grid) * np.cos(phi_grid),
        np.sin(theta_grid) * np.sin(phi_grid),
        np.cos(theta_grid)
    ])
    
    best_points = points.copy()
    best_radius = _covering_radius(points, grid_points)
    
    # Phase 1: Repulsion relaxation
    if time.time() < deadline - 30.0:
        points = _repulsion_relax(points, deadline=deadline - 30.0)
        points = _normalize(points)
        current_radius = _covering_radius(points, grid_points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
    
    # Phase 2: Lloyd's algorithm
    if time.time() < deadline - 20.0:
        for _ in range(50):
            if time.time() > deadline - 20.0:
                break
            points = _lloyd_step_vectorized(points)
            points = _normalize(points)
            current_radius = _covering_radius(points, grid_points)
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = points.copy()
    
    # Phase 3: Local search
    if time.time() < deadline - 10.0:
        points = _local_search(best_points, grid_points, deadline - 10.0)
        current_radius = _covering_radius(points, grid_points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
    
    # Phase 4: Projected gradient polish
    if time.time() < deadline - 5.0:
        points = _projected_gradient_polish(best_points, deadline - 5.0)
        current_radius = _covering_radius(points, grid_points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
    
    return best_points
