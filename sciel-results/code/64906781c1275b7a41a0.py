import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _compute_covering_radius(points):
    """
    Compute the covering radius of a set of points on the unit sphere.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        max_radius = 0.0
        
        for tri in facets:
            p1 = points[tri[0]]
            p2 = points[tri[1]]
            p3 = points[tri[2]]
            
            v1 = p2 - p1
            v2 = p3 - p1
            cross = np.cross(v1, v2)
            norm_cross = np.linalg.norm(cross)
            
            if norm_cross < 1e-12:
                continue
                
            c = cross / norm_cross
            
            if np.dot(c, p1) < 0:
                c = -c
            
            dot_val = np.clip(np.dot(c, p1), -1.0, 1.0)
            dist = np.arccos(dot_val)
            
            if dist > max_radius:
                max_radius = dist
                
    except Exception:
        return np.pi / 2
        
    return np.degrees(max_radius)

def _facet_circumcenters(points, hull):
    """Compute circumcenters (outward normals) and radii for all facets."""
    facets = hull.simplices
    circumcenters = []
    radii = []
    
    for tri in facets:
        p1 = points[tri[0]]
        p2 = points[tri[1]]
        p3 = points[tri[2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross)
        
        if norm_cross < 1e-12:
            continue
            
        c = cross / norm_cross
        
        if np.dot(c, p1) < 0:
            c = -c
        
        dot_val = np.clip(np.dot(c, p1), -1.0, 1.0)
        dist = np.arccos(dot_val)
        
        circumcenters.append(c)
        radii.append(dist)
    
    return np.array(circumcenters), np.array(radii)

def _smooth_max(radii, temperature=0.01):
    """Smooth maximum using log-sum-exp trick."""
    if len(radii) == 0:
        return 0.0
    max_r = np.max(radii)
    exps = np.exp((radii - max_r) / temperature)
    return max_r + temperature * np.log(np.sum(exps))

def _gradient_step(points, step_size, temperature=0.01):
    """
    Compute gradient step to minimize smooth maximum of facet circumradii.
    Returns updated points and the smooth max value.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
    except Exception:
        return points, np.pi / 2
    
    # Compute circumcenters and radii
    circumcenters, radii = _facet_circumcenters(points, hull)
    
    if len(radii) == 0:
        return points, np.pi / 2
    
    # Smooth maximum
    smooth_max_val = _smooth_max(radii, temperature)
    
    # Compute gradient for each point
    # For each facet, the circumradius r satisfies cos(r) = c·p1 = c·p2 = c·p3
    # We want to minimize the smooth max of r
    # Gradient with respect to point i: sum over facets containing i of weight_j * d(r_j)/d(p_i)
    
    # For a facet with circumcenter c and radius r, the derivative of r with respect to a vertex p:
    # r = arccos(c·p), so dr/dp = -c / sqrt(1 - (c·p)^2) = -c / sin(r)
    # But we need to be careful: c depends on the vertices too.
    
    # Simpler approach: use finite differences or a heuristic
    # For now, use the approach of moving vertices toward their facet's circumcenter
    # weighted by how much that facet contributes to the smooth max
    
    # Weight each facet by its contribution to the smooth max
    max_r = np.max(radii)
    weights = np.exp((radii - max_r) / temperature)
    weights /= np.sum(weights)
    
    # For each point, accumulate the weighted direction toward circumcenters of adjacent facets
    gradients = np.zeros_like(points)
    
    for idx, tri in enumerate(facets):
        if idx >= len(weights):
            continue
        w = weights[idx]
        if w < 1e-6:
            continue
        
        c = circumcenters[idx]
        
        for v_idx in tri:
            p = points[v_idx]
            # Direction toward circumcenter
            direction = c - p
            norm_dir = np.linalg.norm(direction)
            if norm_dir > 1e-12:
                direction /= norm_dir
            gradients[v_idx] += w * direction
    
    # Normalize gradients
    grad_norms = np.linalg.norm(gradients, axis=1, keepdims=True)
    grad_norms = np.where(grad_norms < 1e-12, 1.0, grad_norms)
    gradients /= grad_norms
    
    # Apply gradient step
    new_points = points + step_size * gradients
    new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
    
    return new_points, smooth_max_val

def construct():
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    # Phase 1: Gradient descent with smooth max
    step_size = 0.05
    temperature = 0.02
    
    while time.time() < deadline - 20:
        new_points, smooth_val = _gradient_step(points, step_size, temperature)
        
        # Check if valid
        if not np.all(np.isfinite(new_points)):
            break
        
        new_radius = _compute_covering_radius(new_points)
        
        if new_radius < best_radius:
            best_radius = new_radius
            best_points = new_points.copy()
            points = new_points
            step_size *= 0.98
            # Slowly decrease temperature
            temperature *= 0.995
        else:
            # Revert and adjust
            step_size *= 0.9
            if step_size < 0.001:
                # Try random perturbation
                noise = np.random.normal(0, 0.02, (N, 3))
                perturbed = points + noise
                perturbed /= np.linalg.norm(perturbed, axis=1, keepdims=True)
                pert_radius = _compute_covering_radius(perturbed)
                if pert_radius < best_radius:
                    best_radius = pert_radius
                    best_points = perturbed.copy()
                    points = perturbed
                    step_size = 0.01
                else:
                    # Break if stuck
                    break
    
    # Phase 2: Fine-tuning with smaller steps
    step_size = 0.005
    while time.time() < deadline:
        new_points, _ = _gradient_step(points, step_size, 0.005)
        if not np.all(np.isfinite(new_points)):
            break
        
        new_radius = _compute_covering_radius(new_points)
        if new_radius < best_radius:
            best_radius = new_radius
            best_points = new_points.copy()
            points = new_points
            step_size *= 0.98
        else:
            step_size *= 0.9
            if step_size < 1e-5:
                break
    
    return best_points
