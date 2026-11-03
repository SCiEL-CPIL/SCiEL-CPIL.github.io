import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _compute_covering_radius_exact(points):
    """Compute exact covering radius using convex hull."""
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        M = len(facets)
        max_radius = 0.0
        
        for k in range(M):
            tri = facets[k]
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
            dot_val = np.clip(np.dot(c, p1), -1.0, 1.0)
            dist = np.arccos(dot_val)
            max_radius = max(max_radius, dist)
            
        return np.degrees(max_radius)
    except Exception:
        return 180.0

def _soft_max_distance(points, temperature=0.1):
    """
    Compute soft-max of angular distances from each point to nearest other points.
    This serves as a differentiable surrogate for the covering radius.
    """
    # Compute pairwise angular distances
    dot = np.clip(points @ points.T, -1.0, 1.0)
    angular_dist = np.arccos(dot)
    
    # For each point, find the maximum distance to any other point
    # But we want the covering radius, which is about the "holes"
    # A better approach: for each point, compute distance to its farthest neighbor
    # Actually, the covering radius relates to the maximum circumradius of Voronoi cells
    
    # Alternative: use the fact that covering radius ≈ max over points of distance to nearest point
    # But that's the packing radius, not covering radius.
    
    # For covering radius, we need to consider the "gaps" between points.
    # A good surrogate: for each point, compute the distance to the nearest OTHER point,
    # then take a soft-max over all points. But this measures packing, not covering.
    
    # Better: The covering radius is related to the maximum distance from any sphere point
    # to the nearest configuration point. This can be approximated by considering the
    # circumcircles of Delaunay triangles.
    
    # Since we want to avoid convex hull, let's use a different approach:
    # The covering radius is bounded by the maximum distance between neighboring points.
    # For a good configuration, the covering radius is approximately half the maximum
    # nearest-neighbor distance.
    
    # Let's compute: for each point, the distance to its nearest neighbor
    # Then take the max (or soft-max) of these distances
    np.fill_diagonal(angular_dist, np.inf)
    min_dist = np.min(angular_dist, axis=1)  # distance to nearest neighbor for each point
    
    # The covering radius is roughly related to the max of these, but not directly.
    # Actually, for uniform distributions, covering_radius ≈ max_nearest_neighbor / 2 * sqrt(3)
    # But this is an approximation.
    
    # Let's use a different strategy: compute the soft-max of the distances
    # This will be differentiable and guide optimization
    
    # Use log-sum-exp for numerical stability
    scaled = min_dist / temperature
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_max = temperature * log_sum_exp
    
    return soft_max, min_dist

def _gradient_soft_max_distance(points, temperature=0.1):
    """
    Compute gradient of the soft-max distance objective.
    """
    dot = np.clip(points @ points.T, -1.0, 1.0)
    angular_dist = np.arccos(dot)
    np.fill_diagonal(angular_dist, np.inf)
    min_dist = np.min(angular_dist, axis=1)
    
    # For soft-max: f = T * log(sum(exp(d_i / T)))
    # df/dp_j = (1/T) * sum_i exp(d_i/T) / sum_i exp(d_i/T) * dd_i/dp_j
    # where d_i is the min distance for point i
    
    # This is complex. Let's use a simpler gradient:
    # For each point, compute the direction that increases its distance to nearest neighbor
    # This is approximately the direction away from the nearest neighbor
    
    # Compute nearest neighbor for each point
    nearest_idx = np.argmin(angular_dist, axis=1)
    
    # Gradient: for each point, move away from its nearest neighbor
    # But we want to MINIMIZE the max distance, so we want to move points
    # such that the maximum distance decreases.
    
    # Actually, we want to minimize the covering radius, which means
    # we want points to be more uniformly distributed.
    
    # Let's use a different approach: repulsive forces based on soft-max
    # The gradient of the soft-max with respect to point positions can be
    # approximated by weighted repulsion from all other points.
    
    # Compute weights based on soft-max
    scaled = min_dist / temperature
    max_scaled = np.max(scaled)
    weights = np.exp(scaled - max_scaled)
    weights = weights / np.sum(weights)
    
    # For each point, compute the force that would increase its min_dist
    # This is approximately the direction away from the nearest neighbor
    forces = np.zeros_like(points)
    for i in range(N):
        j = nearest_idx[i]
        # Direction from j to i
        direction = points[i] - points[j]
        direction = direction / np.linalg.norm(direction)
        # Weight by the importance of this point in the soft-max
        forces[i] += weights[i] * direction
    
    # Project onto tangent plane
    dot_fn = np.sum(forces * points, axis=1, keepdims=True)
    tangent_forces = forces - dot_fn * points
    
    return tangent_forces

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
    deadline = start_time + 140.0  # 10s safety margin
    
    best_points = points.copy()
    best_radius = _compute_covering_radius_exact(points)
    
    # Optimization parameters
    step_size = 0.05
    min_step_size = 1e-6
    temperature = 0.1
    stagnant_count = 0
    max_stagnant = 30
    
    while time.time() < deadline:
        # Compute soft-max objective and gradient
        soft_max_val, min_dist = _soft_max_distance(points, temperature)
        forces = _gradient_soft_max_distance(points, temperature)
        
        # Update points
        new_points = points + step_size * forces
        new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
        
        # Check for validity
        if not np.all(np.isfinite(new_points)):
            break
        
        points = new_points
        
        # Periodically check exact radius (every 10 iterations or so)
        # to update best and detect stagnation
        stagnant_count += 1
        if stagnant_count % 10 == 0 or time.time() < start_time + 1.0:
            current_radius = _compute_covering_radius_exact(points)
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = points.copy()
                stagnant_count = 0
                # Reduce temperature to focus on max
                temperature = max(temperature * 0.95, 0.01)
            else:
                stagnant_count += 1
                step_size = max(step_size * 0.95, min_step_size)
                
                if stagnant_count > max_stagnant:
                    # Random kick to escape local minimum
                    kick = rng.normal(0, 0.05, points.shape)
                    points += kick
                    points /= np.linalg.norm(points, axis=1, keepdims=True)
                    step_size = 0.05
                    temperature = 0.1
                    stagnant_count = 0
    
    # Final polish with exact method
    if time.time() < deadline - 5.0:
        # Do a few more iterations with exact gradient
        for _ in range(50):
            if time.time() >= deadline - 5.0:
                break
            
            try:
                hull = ConvexHull(points)
                facets = hull.simplices
                M = len(facets)
                centers = np.zeros((M, 3))
                radii = np.zeros(M)
                
                for k in range(M):
                    tri = facets[k]
                    p1 = points[tri[0]]
                    p2 = points[tri[1]]
                    p3 = points[tri[2]]
                    
                    v1 = p2 - p1
                    v2 = p3 - p1
                    cross = np.cross(v1, v2)
                    norm_cross = np.linalg.norm(cross)
                    
                    if norm_cross < 1e-12:
                        radii[k] = 0.0
                        centers[k] = p1
                        continue
                    
                    c = cross / norm_cross
                    dot_val = np.clip(np.dot(c, p1), -1.0, 1.0)
                    radii[k] = np.arccos(dot_val)
                    centers[k] = c
                
                worst_idx = np.argmax(radii)
                worst_center = centers[worst_idx]
                worst_facet = facets[worst_idx]
                
                # Move the vertices of the worst facet toward the center
                for idx in worst_facet:
                    vec = worst_center - points[idx]
                    norm_vec = np.linalg.norm(vec)
                    if norm_vec > 1e-12:
                        points[idx] += 0.01 * vec / norm_vec
                
                # Renormalize
                points /= np.linalg.norm(points, axis=1, keepdims=True)
                
                # Check if improved
                current_radius = _compute_covering_radius_exact(points)
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_points = points.copy()
            except Exception:
                break
    
    return best_points
