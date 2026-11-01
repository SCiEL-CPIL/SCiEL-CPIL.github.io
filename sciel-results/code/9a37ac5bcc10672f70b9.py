"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import ConvexHull
import math

N = 50

# EVOLVE-BLOCK-START
def _compute_covering_radius(points):
    """Compute the covering radius in degrees.
    
    The covering radius is the maximum angular distance from any point on the sphere
    to the nearest configuration point. This equals the maximum circumradius of the
    facets of the convex hull (in angular terms), or equivalently, the maximum 
    angular distance from the origin to the furthest point in the spherical Voronoi 
    diagram.
    
    A practical approach: compute the convex hull, then for each facet (triangle),
    find the angular distance from the origin to the plane of the facet. The covering
    radius is the maximum of these distances.
    """
    points = np.array(points)
    n = len(points)
    
    # Compute convex hull
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: if hull fails, estimate using nearest neighbor distances
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        # The covering radius is bounded by half the maximum nearest-neighbor distance
        # but this is an overestimate. Use a better estimate.
        # Actually, let's use the angular diameter of the largest empty circle
        # For now, return a large value to indicate failure
        return 180.0
    
    # For each facet, compute the angular distance from origin to the facet plane
    # The distance from origin to a plane with normal n and offset d is |d|/||n||
    # The angular distance is arccos(d / ||n||) where the plane equation is n·x = d
    # For a facet with vertices v1, v2, v3, the normal is (v2-v1) × (v3-v1)
    # The plane equation: n·x = n·v1
    # Distance from origin: |n·v1| / ||n||
    # Since points are on unit sphere and origin is inside, n·v1 should be positive
    # Angular distance: arccos(|n·v1| / ||n||)
    
    max_ang_dist = 0.0
    for simplex in hull.simplices:
        v1 = points[simplex[0]]
        v2 = points[simplex[1]]
        v3 = points[simplex[2]]
        
        n = np.cross(v2 - v1, v3 - v1)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-15:
            continue
        
        # Plane equation: n · x = n · v1
        d = np.dot(n, v1)
        
        # The angular distance from origin to this plane is arccos(d / n_norm)
        # But we need to be careful: the covering radius is the maximum angular distance
        # from any point on the sphere to the nearest configuration point.
        # For a facet, the furthest point on the sphere from all three vertices is the
        # point on the sphere closest to the facet plane (in the direction away from origin).
        # Actually, the covering radius is determined by the "empty caps" - the largest
        # empty spherical cap. For each facet, the center of the largest empty cap is
        # the point on the sphere in the direction of the facet normal (away from origin).
        # The angular distance from this center to any of the three vertices is the
        # circumradius of that facet in angular terms.
        
        # The point on the sphere in the direction of n (normalized) is the center.
        n_unit = n / n_norm
        # Angular distance from n_unit to v1:
        cos_ang = np.dot(n_unit, v1)
        cos_ang = np.clip(cos_ang, -1.0, 1.0)
        ang = np.arccos(cos_ang)
        
        if ang > max_ang_dist:
            max_ang_dist = ang
    
    return np.degrees(max_ang_dist)


def _lloyd_relaxation(points, max_iter=200, tol=1e-10):
    """Lloyd's algorithm for spherical k-means / covering.
    
    Iteratively:
    1. Compute spherical Voronoi regions (approximated via convex hull facets)
    2. Move each point to the spherical centroid of its Voronoi region
    """
    points = np.array(points, dtype=np.float64)
    n = len(points)
    
    for iteration in range(max_iter):
        # Compute convex hull
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        # For each point, find its associated facets
        # Each point is a vertex of the hull, and its Voronoi region is the union
        # of all facets adjacent to it.
        # The spherical centroid of the Voronoi region can be approximated by
        # the average of the facet normals (weighted by facet area) for adjacent facets.
        
        # Build vertex-to-facets mapping
        vertex_facets = [[] for _ in range(n)]
        for i, simplex in enumerate(hull.simplices):
            for v in simplex:
                vertex_facets[v].append(i)
        
        new_points = np.zeros_like(points)
        for i in range(n):
            if not vertex_facets[i]:
                new_points[i] = points[i]
                continue
            
            # Compute weighted average of facet normals
            total_weight = 0.0
            centroid = np.zeros(3)
            for facet_idx in vertex_facets[i]:
                simplex = hull.simplices[facet_idx]
                v1 = points[simplex[0]]
                v2 = points[simplex[1]]
                v3 = points[simplex[2]]
                
                n_vec = np.cross(v2 - v1, v3 - v1)
                n_norm = np.linalg.norm(n_vec)
                if n_norm < 1e-15:
                    continue
                
                # Facet area (on the sphere, this is related to the solid angle)
                # Use the magnitude of the cross product as a proxy for area
                weight = n_norm
                
                centroid += n_vec * weight
                total_weight += weight
            
            if total_weight > 1e-15:
                centroid /= total_weight
                norm = np.linalg.norm(centroid)
                if norm > 1e-15:
                    new_points[i] = centroid / norm
                else:
                    new_points[i] = points[i]
            else:
                new_points[i] = points[i]
        
        # Check convergence
        diff = np.linalg.norm(new_points - points)
        points = new_points
        
        if diff < tol:
            break
    
    return points


def _generate_initial_configurations():
    """Generate multiple initial configurations."""
    configs = []
    
    # 1. Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    configs.append(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))
    
    # 2. Random points with rejection sampling for minimum separation
    rng = np.random.default_rng(42)
    for trial in range(5):
        pts = []
        min_sep = 0.3  # radians
        for _ in range(N):
            for attempt in range(1000):
                p = rng.normal(size=3)
                p /= np.linalg.norm(p)
                if all(np.arccos(np.clip(np.dot(p, q), -1, 1)) > min_sep for q in pts):
                    pts.append(p)
                    break
            else:
                p = rng.normal(size=3)
                p /= np.linalg.norm(p)
                pts.append(p)
        configs.append(np.array(pts))
    
    # 3. Icosahedron-based expansion
    # Start with icosahedron (12 vertices) and subdivide
    # This is complex, so use a different approach:
    # Use a grid-based initialization
    latitudes = np.linspace(-np.pi/2, np.pi/2, 8)
    longitudes = np.linspace(0, 2*np.pi, 8, endpoint=False)
    pts = []
    for lat in latitudes:
        for lon in longitudes:
            x = np.cos(lat) * np.cos(lon)
            y = np.cos(lat) * np.sin(lon)
            z = np.sin(lat)
            pts.append([x, y, z])
    # We have 64 points, need 50. Keep the first 50.
    configs.append(np.array(pts[:N]))
    
    return configs


def construct():
    """Construct the optimal covering configuration."""
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # Generate initial configurations
    configs = _generate_initial_configurations()
    
    best_points = None
    best_radius = float('inf')
    
    for init_idx, init_pts in enumerate(configs):
        if time.time() > deadline:
            break
        
        # Run Lloyd relaxation
        relaxed = _lloyd_relaxation(init_pts, max_iter=300, tol=1e-12)
        
        # Compute covering radius
        radius = _compute_covering_radius(relaxed)
        
        if radius < best_radius:
            best_radius = radius
            best_points = relaxed.copy()
    
    # If we have time, try a gradient-based refinement
    if best_points is not None and time.time() < deadline - 20.0:
        # Simple gradient descent on the covering radius
        # The covering radius is the max over facets of arccos(n·v1/||n||)
        # We can use a soft-max approximation for smoothness
        for grad_iter in range(100):
            if time.time() > deadline - 5.0:
                break
            
            try:
                hull = ConvexHull(best_points)
            except Exception:
                break
            
            # Compute gradients for each point
            grad = np.zeros_like(best_points)
            
            # For each facet, compute the angular distance and its gradient
            # The angular distance for a facet with vertices i,j,k is:
            # arccos(n·vi / ||n||) where n = (vj-vi)×(vk-vi)
            # This is complex to differentiate. Instead, use a simpler approach:
            # Move each point slightly toward the centroid of its adjacent facet centers.
            
            vertex_facets = [[] for _ in range(N)]
            for i, simplex in enumerate(hull.simplices):
                for v in simplex:
                    vertex_facets[v].append(i)
            
            for i in range(N):
                if not vertex_facets[i]:
                    continue
                
                # Compute the "pressure" on this vertex: the average of the 
                # outward normals of adjacent facets, weighted by the angular distance
                total_force = np.zeros(3)
                for facet_idx in vertex_facets[i]:
                    simplex = hull.simplices[facet_idx]
                    v1 = best_points[simplex[0]]
                    v2 = best_points[simplex[1]]
                    v3 = best_points[simplex[2]]
                    
                    n_vec = np.cross(v2 - v1, v3 - v1)
                    n_norm = np.linalg.norm(n_vec)
                    if n_norm < 1e-15:
                        continue
                    
                    n_unit = n_vec / n_norm
                    cos_ang = np.clip(np.dot(n_unit, v1), -1, 1)
                    ang = np.arccos(cos_ang)
                    
                    # The force should push the vertex to reduce the angular distance
                    # This is approximately in the direction that reduces the circumradius
                    total_force += n_unit * ang * n_norm
                
                if np.linalg.norm(total_force) > 1e-15:
                    # Project onto the tangent plane
                    tangent = total_force - np.dot(total_force, best_points[i]) * best_points[i]
                    grad[i] = -tangent  # Negative gradient direction
            
            # Take a small step
            step_size = 0.01
            new_points = best_points + step_size * grad
            # Re-normalize
            for i in range(N):
                norm = np.linalg.norm(new_points[i])
                if norm > 1e-15:
                    new_points[i] /= norm
            
            # Check if the new configuration is valid
            new_radius = _compute_covering_radius(new_points)
            if new_radius < best_radius:
                best_points = new_points
                best_radius = new_radius
            else:
                # Reduce step size
                step_size *= 0.5
    
    if best_points is None:
        # Fallback to Fibonacci spiral
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Ensure all points are on the unit sphere
    for i in range(N):
        norm = np.linalg.norm(best_points[i])
        if norm > 1e-15:
            best_points[i] /= norm
    
    return best_points
# EVOLVE-BLOCK-END
