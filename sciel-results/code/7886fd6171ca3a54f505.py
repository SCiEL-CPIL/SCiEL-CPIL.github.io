"""Model-and-optimize approach for covering the sphere with 130 equal caps."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 130

def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return points / norms

def _covering_radius(points):
    """
    Compute the covering radius in radians.
    points: (N, 3) unit vectors.
    The covering radius is the maximum over all points y on the sphere
    of the angular distance to the nearest point in the configuration.
    
    This equals the maximum circumradius of the facets of the convex hull.
    For each triangular facet, we compute the circumcenter and check if it's
    on the correct side (empty cap). The circumradius is the angular distance
    from the circumcenter to any vertex of the triangle.
    """
    points = _normalize(points)
    
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: use a grid-based approximation
        # Create a fine grid on the sphere
        theta = np.linspace(0, np.pi, 200)
        phi = np.linspace(0, 2 * np.pi, 400)
        theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
        grid_points = np.column_stack([
            np.sin(theta_grid) * np.cos(phi_grid),
            np.sin(theta_grid) * np.sin(phi_grid),
            np.cos(theta_grid)
        ]).reshape(-1, 3)
        
        # For each grid point, find the nearest configuration point
        max_dist = 0.0
        for y in grid_points:
            dots = points @ y
            dots = np.clip(dots, -1.0, 1.0)
            min_angle = np.arccos(dots.max())
            if min_angle > max_dist:
                max_dist = min_angle
        return max_dist
    
    max_radius = 0.0
    for simplex in hull.simplices:
        # Get the three vertices of the triangle
        v0, v1, v2 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        
        # Compute the circumcenter of the triangle on the sphere
        # The circumcenter is the point equidistant from all three vertices
        # Solve for c such that c·v0 = c·v1 = c·v2, |c| = 1
        
        # Using the formula for circumcenter of a spherical triangle
        # The circumcenter lies in the plane of the triangle
        # c = a*v0 + b*v1 + d*v2 where a, b, d are coefficients
        
        # Alternative: use the fact that the circumcenter is perpendicular to 
        # the edges in the tangent plane
        
        # Let's use a more robust method: find the point c on the sphere
        # that maximizes the minimum distance to the three vertices, subject to
        # c being in the convex hull of the triangle (i.e., in the correct cap)
        
        # Actually, the circumcenter of a spherical triangle can be computed as:
        # c = normalize(v0 + v1 + v2) if the triangle is small, but this is not exact.
        
        # More exact method: the circumcenter is the intersection of the sphere
        # with the line through the centroid perpendicular to the plane of the triangle.
        
        # Let's compute the normal to the plane of the triangle
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-10:
            continue
        normal = normal / normal_norm
        
        # The circumcenter is along the normal direction from the centroid
        centroid = (v0 + v1 + v2) / 3.0
        
        # We need to find t such that |centroid + t*normal| = 1
        # |centroid|^2 + 2*t*(centroid·normal) + t^2 = 1
        c0 = centroid @ normal
        c2 = 1.0 - np.dot(centroid, centroid)
        # t^2 + 2*c0*t + c2 = 0
        discriminant = c0*c0 - c2
        if discriminant < 0:
            # No real solution, use the centroid normalized
            c = centroid / np.linalg.norm(centroid)
        else:
            sqrt_disc = np.sqrt(discriminant)
            t1 = -c0 + sqrt_disc
            t2 = -c0 - sqrt_disc
            
            # Choose the t that gives a point on the correct side
            c1 = centroid + t1 * normal
            c2_point = centroid + t2 * normal
            
            # Check which one is in the correct cap (same side as the triangle)
            # The correct circumcenter should be on the same side as the normal
            # pointing outward from the convex hull
            
            # For a convex hull, the outward normal is the one that points away from the origin
            # The triangle's normal should point outward
            if np.dot(normal, centroid) > 0:
                # Normal points outward, so we want the circumcenter on the same side
                # Both c1 and c2 should be checked
                c = c1 if np.linalg.norm(c1) > np.linalg.norm(c2_point) else c2_point
            else:
                c = c2_point if np.linalg.norm(c2_point) > np.linalg.norm(c1) else c1
        
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-10:
            continue
        c = c / c_norm
        
        # Check if the cap is empty: the circumcenter should be farther from all
        # other points than from the triangle vertices
        dots = points @ c
        dots = np.clip(dots, -1.0, 1.0)
        max_dot = dots.max()
        
        # The circumradius is the angular distance from c to any vertex
        dot_v0 = np.clip(np.dot(c, v0), -1.0, 1.0)
        radius = np.arccos(dot_v0)
        
        # Check if this facet has an empty cap
        # The cap is empty if no other point is closer to c than the vertices
        # i.e., max_dot < dot_v0 (approximately)
        if max_dot <= dot_v0 + 1e-10:
            if radius > max_radius:
                max_radius = radius
    
    return max_radius

def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using the Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _local_optimization(points, max_iter=500, tol=1e-8):
    """
    Local optimization using gradient descent to minimize the covering radius.
    """
    points = _normalize(points).copy()
    
    for iteration in range(max_iter):
        # Compute the current covering radius and find the critical facet
        hull = ConvexHull(points)
        max_radius = 0.0
        best_simplex = None
        
        for simplex in hull.simplices:
            v0, v1, v2 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            normal_norm = np.linalg.norm(normal)
            if normal_norm < 1e-10:
                continue
            normal = normal / normal_norm
            
            centroid = (v0 + v1 + v2) / 3.0
            c0 = centroid @ normal
            c2 = 1.0 - np.dot(centroid, centroid)
            discriminant = c0*c0 - c2
            
            if discriminant < 0:
                c = centroid / np.linalg.norm(centroid)
            else:
                sqrt_disc = np.sqrt(discriminant)
                t1 = -c0 + sqrt_disc
                t2 = -c0 - sqrt_disc
                c1 = centroid + t1 * normal
                c2_point = centroid + t2 * normal
                c = c1 if np.linalg.norm(c1) > np.linalg.norm(c2_point) else c2_point
            
            c_norm = np.linalg.norm(c)
            if c_norm < 1e-10:
                continue
            c = c / c_norm
            
            dots = points @ c
            dots = np.clip(dots, -1.0, 1.0)
            max_dot = dots.max()
            
            dot_v0 = np.clip(np.dot(c, v0), -1.0, 1.0)
            radius = np.arccos(dot_v0)
            
            if max_dot <= dot_v0 + 1e-10 and radius > max_radius:
                max_radius = radius
                best_simplex = simplex
        
        if best_simplex is None:
            break
        
        v0, v1, v2 = points[best_simplex[0]], points[best_simplex[1]], points[best_simplex[2]]
        
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-10:
            break
        normal = normal / normal_norm
        
        centroid = (v0 + v1 + v2) / 3.0
        c0 = centroid @ normal
        c2 = 1.0 - np.dot(centroid, centroid)
        discriminant = c0*c0 - c2
        
        if discriminant < 0:
            c = centroid / np.linalg.norm(centroid)
        else:
            sqrt_disc = np.sqrt(discriminant)
            t1 = -c0 + sqrt_disc
            t2 = -c0 - sqrt_disc
            c1 = centroid + t1 * normal
            c2_point = centroid + t2 * normal
            c = c1 if np.linalg.norm(c1) > np.linalg.norm(c2_point) else c2_point
        
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-10:
            break
        c = c / c_norm
        
        # Move the three vertices towards the circumcenter to reduce the radius
        # The gradient of the radius with respect to each vertex points away from c
        # So we move each vertex towards c
        
        step_size = 0.01
        for idx in best_simplex:
            # Move point idx towards c
            direction = c - points[idx]
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 1e-10:
                points[idx] += step_size * direction / direction_norm
        
        points = _normalize(points)
        
        # Check convergence
        if iteration > 0 and abs(max_radius - prev_radius) < tol:
            break
        prev_radius = max_radius
    
    return points

def construct():
    """
    Construct 130 points on the unit sphere minimizing the covering radius.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Start with Fibonacci spiral
    points = _fibonacci_sphere(N)
    
    # Perform local optimization
    points = _local_optimization(points, max_iter=100)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time - 10:
        return _normalize(points)
    
    # Try a few random restarts if time permits
    best_points = points
    best_radius = _covering_radius(points)
    
    rng = np.random.default_rng(42)
    while time.time() < deadline - 20:
        # Generate random initial points
        random_points = rng.standard_normal((N, 3))
        random_points = _normalize(random_points)
        
        # Local optimization
        optimized = _local_optimization(random_points, max_iter=50)
        radius = _covering_radius(optimized)
        
        if radius < best_radius:
            best_radius = radius
            best_points = optimized
    
    return _normalize(best_points)
