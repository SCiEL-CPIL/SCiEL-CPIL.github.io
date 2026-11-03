import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _compute_circumradii(points, hull):
    """
    Compute circumradii and circumcenters for all hull facets.
    Returns:
        radii: (M,) array of circumradii (angular)
        centers: (M, 3) array of circumcenters on sphere
        facets: (M, 3) array of facet indices
    """
    facets = hull.simplices
    M = len(facets)
    
    p1 = points[facets[:, 0]]
    p2 = points[facets[:, 1]]
    p3 = points[facets[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = np.cross(v1, v2)
    cross_sq = np.einsum('ij,ij->i', cross, cross)
    
    valid = cross_sq > 1e-28
    centers = np.zeros((M, 3))
    radii = np.zeros(M)
    
    if np.any(valid):
        # Circumcenter via: p1 + t * cross, where t = (cross . p1) / (cross . cross)
        # This gives the 3D circumcenter; normalize to get spherical circumcenter
        dot_cp = np.einsum('ij,ij->i', cross[valid], p1[valid])
        t = dot_cp / cross_sq[valid]
        c_3d = p1[valid] + t[:, np.newaxis] * cross[valid]
        
        # Normalize to unit sphere
        norms = np.linalg.norm(c_3d, axis=1, keepdims=True)
        norms[norms < 1e-14] = 1.0
        c = c_3d / norms
        
        # Ensure centers point to the empty cap side (same side as centroid)
        centroids = (p1[valid] + p2[valid] + p3[valid]) / 3.0
        dots = np.einsum('ij,ij->i', c, centroids)
        flip = dots < 0
        c[flip] = -c[flip]
        
        centers[valid] = c
        
        # Compute angular distances
        dot_vals = np.einsum('ij,ij->i', c, p1[valid])
        dot_vals = np.clip(dot_vals, -1.0, 1.0)
        radii[valid] = np.arccos(dot_vals)
    
    return radii, centers, facets

def _compute_covering_radius(points):
    """
    Compute the covering radius in degrees.
    """
    try:
        hull = ConvexHull(points)
        radii, centers, facets = _compute_circumradii(points, hull)
        return np.degrees(np.max(radii))
    except Exception:
        return 180.0

def _soft_max_gradient(points, hull, temperature=0.01):
    """
    Compute a smooth gradient using soft-max weighting of all facets.
    This creates a smooth surrogate for the nonsmooth max objective.
    
    For each facet, compute weight = exp(radii[k] / temperature) / sum(exp(radii[j] / temperature))
    Then for each vertex of each facet, add the direction toward the circumcenter
    weighted by the facet's soft-max weight and the facet's radius.
    
    Returns:
        gradient: (N, 3) array of gradient directions
    """
    radii, centers, facets = _compute_circumradii(points, hull)
    
    # Soft-max weights: emphasizes facets near the maximum
    # Use temperature to control smoothness
    scaled_radii = radii / temperature
    # Subtract max for numerical stability
    scaled_radii -= np.max(scaled_radii)
    exp_weights = np.exp(scaled_radii)
    weights = exp_weights / (np.sum(exp_weights) + 1e-30)
    
    # Scale weights by radius to prioritize larger holes
    weights *= radii
    
    gradient = np.zeros_like(points)
    
    # For each facet, distribute force to its three vertices
    for k in range(len(facets)):
        if weights[k] < 1e-10:
            continue
        
        c = centers[k]
        tri = facets[k]
        w = weights[k]
        
        for idx in tri:
            p = points[idx]
            # Direction from p to c, projected onto tangent plane
            v = c - p
            dot_vp = np.dot(v, p)
            tangent_v = v - dot_vp * p
            norm_tv = np.linalg.norm(tangent_v)
            if norm_tv > 1e-14:
                gradient[idx] += w * tangent_v / norm_tv
    
    return gradient

def _per_point_max_gradient(points, hull):
    """
    For each point, find the adjacent facet with maximum circumradius,
    and move the point toward that facet's circumcenter.
    Scale by the facet's radius to prioritize large holes.
    """
    radii, centers, facets = _compute_circumradii(points, hull)
    
    # For each point, track the max radius and corresponding center
    point_max_radius = np.zeros(N)
    point_max_center = np.zeros((N, 3))
    
    for k in range(len(facets)):
        r_k = radii[k]
        if r_k <= 0:
            continue
        c_k = centers[k]
        tri = facets[k]
        for idx in tri:
            if r_k > point_max_radius[idx]:
                point_max_radius[idx] = r_k
                point_max_center[idx] = c_k
    
    gradient = np.zeros_like(points)
    
    for i in range(N):
        if point_max_radius[i] > 0:
            v = point_max_center[i] - points[i]
            dot_vp = np.dot(v, points[i])
            tangent_v = v - dot_vp * points[i]
            norm_tv = np.linalg.norm(tangent_v)
            if norm_tv > 1e-14:
                # Scale by radius to prioritize large holes
                gradient[i] = (tangent_v / norm_tv) * point_max_radius[i]
    
    return gradient

def _compute_circumradii_fast(points, hull):
    """
    Fast vectorized circumradius computation for scoring.
    """
    radii, centers, facets = _compute_circumradii(points, hull)
    return radii

def construct():
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    # Add small random perturbation to break symmetry
    rng = np.random.default_rng(42)
    points += rng.normal(0, 0.01, points.shape)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s limit
    
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    # Phase 1: Coarse optimization with per-point max gradient
    step_size = 0.05
    min_step_size = 1e-7
    no_improve_count = 0
    max_no_improve = 40
    phase1_deadline = start_time + 60.0
    
    while time.time() < phase1_deadline:
        try:
            hull = ConvexHull(points)
        except Exception:
            kick = rng.normal(0, 0.02, points.shape)
            points += kick
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            continue
        
        grad = _per_point_max_gradient(points, hull)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-14:
            break
        
        new_points = points + step_size * grad
        new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
        
        current_radius = _compute_covering_radius(new_points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = new_points.copy()
            points = new_points
            no_improve_count = 0
            step_size = min(step_size * 1.03, 0.1)
        else:
            step_size = max(step_size * 0.85, min_step_size)
            no_improve_count += 1
            
            if no_improve_count > max_no_improve:
                kick = rng.normal(0, 0.03, points.shape)
                points += kick
                points /= np.linalg.norm(points, axis=1, keepdims=True)
                step_size = 0.02
                no_improve_count = 0
                current_radius = _compute_covering_radius(points)
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_points = points.copy()
    
    # Phase 2: Fine optimization with soft-max gradient (smooth surrogate)
    step_size = 0.01
    temperature = 0.02
    no_improve_count = 0
    max_no_improve = 30
    phase2_deadline = start_time + 110.0
    
    while time.time() < phase2_deadline:
        try:
            hull = ConvexHull(points)
        except Exception:
            kick = rng.normal(0, 0.01, points.shape)
            points += kick
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            continue
        
        grad = _soft_max_gradient(points, hull, temperature)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-14:
            break
        
        new_points = points + step_size * grad
        new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
        
        current_radius = _compute_covering_radius(new_points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = new_points.copy()
            points = new_points
            no_improve_count = 0
            step_size = min(step_size * 1.02, 0.02)
        else:
            step_size = max(step_size * 0.9, min_step_size)
            no_improve_count += 1
            
            if no_improve_count > max_no_improve:
                # Decrease temperature to sharpen the soft-max
                temperature = max(temperature * 0.8, 0.005)
                kick = rng.normal(0, 0.01, points.shape)
                points += kick
                points /= np.linalg.norm(points, axis=1, keepdims=True)
                step_size = 0.005
                no_improve_count = 0
                current_radius = _compute_covering_radius(points)
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_points = points.copy()
    
    # Phase 3: Final polish with very small steps and tight tolerance
    step_size = 0.001
    temperature = 0.005
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        grad = _soft_max_gradient(points, hull, temperature)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-14:
            break
        
        new_points = points + step_size * grad
        new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
        
        current_radius = _compute_covering_radius(new_points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = new_points.copy()
            points = new_points
            no_improve_count = 0
        else:
            step_size = max(step_size * 0.95, 1e-8)
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
    
    return best_points
