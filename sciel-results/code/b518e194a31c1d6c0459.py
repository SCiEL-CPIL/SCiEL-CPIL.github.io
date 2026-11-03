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
            
            dot_val = np.clip(np.dot(c, p1), -1.0, 1.0)
            dist = np.arccos(dot_val)
            
            if dist > np.pi / 2:
                dist = np.pi - dist
                
            if dist > max_radius:
                max_radius = dist
                
    except Exception:
        return 180.0
        
    return np.degrees(max_radius)

def _get_circumcenters_and_radii(points, hull):
    """
    Compute circumcenters and radii for all hull facets.
    Returns centers (M, 3), radii (M,), and facets (M, 3).
    """
    facets = hull.simplices
    M = len(facets)
    centers = np.zeros((M, 3))
    radii = np.zeros(M)
    
    for i, tri in enumerate(facets):
        p1 = points[tri[0]]
        p2 = points[tri[1]]
        p3 = points[tri[2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross)
        
        if norm_cross < 1e-12:
            centers[i] = p1
            radii[i] = 0.0
            continue
            
        c = cross / norm_cross
        
        # Circumcenter is along the normal from the origin
        # The distance from origin to the plane is h = |dot(c, p1)|
        h = abs(np.dot(c, p1))
        
        # If the facet is degenerate or very small, handle carefully
        # The circumradius r_circ satisfies r_circ^2 = h^2 + d^2 where d is distance from foot to vertex
        # But for the angular distance, we want the angle from origin to the circumcenter direction
        # The angular distance is arccos(h) if h <= 1, but we need to be careful about the hemisphere
        
        # Actually, the covering radius for a facet is the maximum angular distance from any point on the sphere
        # in that facet's cap to the nearest vertex. This is equal to the circumradius of the spherical triangle.
        
        # For a spherical triangle with vertices at angular distances, the circumradius can be computed.
        # But a simpler approach: the "circumcenter" of the facet on the sphere is the point on the normal
        # (extended to the sphere) that is equidistant (angularly) from the three vertices.
        
        # The point on the sphere along the normal c is c itself (since c is unit).
        # The angular distance from c to p1 is arccos(dot(c, p1)).
        # Due to symmetry, this should be the same for p2 and p3.
        
        dot_val = np.clip(np.dot(c, p1), -1.0, 1.0)
        ang_dist = np.arccos(dot_val)
        
        # If the normal points away from the origin relative to the facet center, we might need to flip
        # Check: the centroid of the triangle
        centroid = (p1 + p2 + p3) / 3.0
        centroid /= np.linalg.norm(centroid)
        
        # If dot(c, centroid) < 0, the normal c points away from the facet's "center" direction
        # In that case, the circumcenter on the sphere should be -c
        if np.dot(c, centroid) < 0:
            c = -c
            dot_val = np.clip(np.dot(c, p1), -1.0, 1.0)
            ang_dist = np.arccos(dot_val)
            
        centers[i] = c
        radii[i] = ang_dist
        
    return centers, radii, facets

def _optimize(points, deadline):
    """
    Optimize points to minimize covering radius using gradient descent
    on the maximum circumradius, with smoothing.
    """
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    # Use a smooth max approximation for the objective
    # f(points) = smooth_max(radii) where radii are the circumradii of facets
    # Gradient of smooth max: sum_i exp(r_i / T) / sum_j exp(r_j / T) * grad_r_i
    
    T = 0.05  # Temperature for smooth max
    step_size = 0.005
    min_step = 1e-6
    
    # Precompute initial hull
    try:
        hull = ConvexHull(points)
    except Exception:
        return best_points
    
    centers, radii, facets = _get_circumcenters_and_radii(points, hull)
    
    # Gradient of the covering radius with respect to points
    # For each facet, the circumradius r_i depends on the three vertices.
    # We use a finite-difference-free approach:
    # The gradient of r_i with respect to vertex p_k (k in facet i) can be derived.
    # For a spherical triangle, the circumradius R satisfies:
    # sin(R) = sin(a) * sin(b) * sin(c) / (2 * area_spherical) ... this is complex.
    # 
    # Simpler: use the fact that the circumcenter c_i is the unit normal to the facet.
    # r_i = arccos(dot(c_i, p_k)) for any vertex k in the facet.
    # dr_i/dp_k = -c_i / sqrt(1 - dot(c_i, p_k)^2) projected onto tangent plane
    # But this is only valid if the circumcenter is on the same side.
    
    # Let's use a different approach: minimize the sum of squared deviations from the max radius
    # This is equivalent to making all radii equal (balanced), which is a necessary condition for optimality.
    
    # Objective: sum_i (r_i - r_max)^2 + lambda * r_max
    # Gradient: for each point, sum over facets containing it of 2*(r_i - r_max) * dr_i/dp
    
    # For simplicity, use a repulsive force between points that are too close in the Voronoi sense
    # and an attractive force to spread them out.
    
    # Actually, let's use a more direct approach:
    # Minimize f = sum_i r_i^2 (sum of squared circumradii)
    # This is a smooth objective that, when minimized, tends to equalize the radii.
    
    # Gradient of sum r_i^2 with respect to point p_j:
    # sum over facets i containing j: 2 * r_i * dr_i/dp_j
    
    # dr_i/dp_j for a facet with circumcenter c_i:
    # r_i = arccos(dot(c_i, p_j))
    # dr_i/dp_j = -c_i / sqrt(1 - dot(c_i, p_j)^2)
    # But we need to project onto the tangent plane at p_j.
    
    iteration = 0
    while time.time() < deadline:
        try:
            hull = ConvexHull(points)
        except Exception:
            # Perturb and retry
            rng = np.random.default_rng(iteration)
            points += rng.normal(0, 0.01, points.shape)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            continue
            
        centers, radii, facets = _get_circumcenters_and_radii(points, hull)
        
        # Compute gradient
        grad = np.zeros_like(points)
        
        for i in range(len(facets)):
            r_i = radii[i]
            c_i = centers[i]
            
            for k in range(3):
                idx = facets[i][k]
                p_k = points[idx]
                
                dot_val = np.dot(c_i, p_k)
                dot_val = np.clip(dot_val, -1.0 + 1e-10, 1.0 - 1e-10)
                denom = np.sqrt(1.0 - dot_val * dot_val)
                
                if denom < 1e-10:
                    continue
                    
                # dr_i/dp_k = -c_i / denom
                # Project onto tangent plane: subtract (dot(grad, p_k)) * p_k
                raw_grad = -c_i / denom
                proj = raw_grad - np.dot(raw_grad, p_k) * p_k
                grad[idx] += 2.0 * r_i * proj
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
            
        # Update
        new_points = points - step_size * grad / grad_norm
        new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
        
        # Check validity
        if not np.all(np.isfinite(new_points)):
            step_size *= 0.5
            continue
            
        # Compute new radius
        new_radius = _compute_covering_radius(new_points)
        
        if new_radius < best_radius:
            best_radius = new_radius
            best_points = new_points.copy()
            points = new_points
            # Slightly increase step size if making progress
            step_size = min(step_size * 1.01, 0.01)
        else:
            # Decay step size
            step_size = max(step_size * 0.95, min_step)
            
        iteration += 1
        
        # Adaptive temperature for smooth max (not used directly, but can be added)
        # T = max(T * 0.99, 0.01)
    
    return best_points

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
    deadline = start_time + 130.0  # Safety margin for 150s limit
    
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    # Run optimization
    optimized_points = _optimize(points, deadline)
    optimized_radius = _compute_covering_radius(optimized_points)
    
    if optimized_radius < best_radius:
        best_points = optimized_points
        best_radius = optimized_radius
    
    return best_points
