"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time

def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.zeros_like(v)
    return v / n

def _spherical_centroid(points):
    """Compute the spherical centroid of a set of points on the sphere."""
    # For points on the unit sphere, the spherical centroid is the normalized sum
    # if they are contained in an open hemisphere. Otherwise, it's more complex.
    # Given the context of Voronoi cells on a sphere with reasonable coverage,
    # they should be in a hemisphere.
    c = np.sum(points, axis=0)
    n = np.linalg.norm(c)
    if n < 1e-12:
        # Fallback: return the mean of the points normalized (rough estimate)
        return _normalize(np.mean(points, axis=0))
    return c / n

def _get_voronoi_centroids(points):
    """
    Compute the spherical centroids of the Voronoi cells for the given points.
    Uses the convex hull of the points to identify facets.
    Each facet of the convex hull corresponds to a Voronoi cell.
    The vertices of the facet are the neighbors of the point.
    The centroid of the facet (in 3D) is not the spherical centroid.
    Instead, for each facet, we identify the point that is the "owner" of the facet.
    Actually, the standard way is:
    1. Compute ConvexHull of points.
    2. For each facet (simplex), the vertices are indices of points.
    3. Wait, the Voronoi cell of point i is the set of points closer to i than to any other.
    4. The vertices of the Voronoi cell of i are the circumcenters of the facets of the convex hull that contain i?
    5. No. The vertices of the Voronoi cell of i are the points equidistant to i and its neighbors.
    6. A more robust method for spherical Voronoi centroids:
       For each point i, find all facets of the convex hull that have i as a vertex.
       The vertices of the Voronoi cell of i are the circumcenters of these facets.
       Then the spherical centroid is the normalized sum of these circumcenters.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return points.copy() # Fallback if hull fails

    # Map from point index to list of facet indices where it appears
    point_to_facets = {i: [] for i in range(len(points))}
    for fac_idx, fac in enumerate(hull.simplices):
        for v in fac:
            point_to_facets[v].append(fac_idx)
    
    centroids = np.zeros_like(points)
    for i in range(len(points)):
        facets = point_to_facets[i]
        if not facets:
            centroids[i] = points[i]
            continue
        
        # Collect circumcenters of these facets
        circumcenters = []
        for f_idx in facets:
            simplex = hull.simplices[f_idx]
            # Get the points forming the simplex
            simplex_pts = points[simplex]
            # Compute circumcenter of the simplex
            # For a triangle (2-simplex), the circumcenter is the intersection of perpendicular bisectors.
            # For a tetrahedron (3-simplex), it's similar.
            # Since we are in 3D, simplices are triangles (2D) or tetrahedra (3D).
            # ConvexHull in 3D returns simplices of dimension d-1, so triangles (3 vertices).
            # Wait, if points are in general position in 3D, the hull facets are triangles.
            # So simplex has 3 vertices.
            if len(simplex) == 3:
                a, b, c = simplex_pts
                # Circumcenter of triangle abc
                # Solve |x-a|^2 = |x-b|^2 = |x-c|^2
                # 2x.(b-a) = |b|^2 - |a|^2
                # 2x.(c-a) = |c|^2 - |a|^2
                # Since points are on unit sphere, |a|^2=|b|^2=|c|^2=1, so RHS is 0.
                # x.(b-a) = 0
                # x.(c-a) = 0
                # So x is orthogonal to (b-a) and (c-a).
                # x is parallel to (b-a) x (c-a)
                v1 = b - a
                v2 = c - a
                cross = np.cross(v1, v2)
                n = np.linalg.norm(cross)
                if n > 1e-12:
                    circumcenters.append(cross / n)
            elif len(simplex) == 4:
                # Tetrahedron circumcenter
                a, b, c, d = simplex_pts
                # |x-a|^2 = |x-b|^2 => 2x.(b-a) = 0
                # |x-a|^2 = |x-c|^2 => 2x.(c-a) = 0
                # |x-a|^2 = |x-d|^2 => 2x.(d-a) = 0
                # x is orthogonal to b-a, c-a, d-a
                # x is parallel to (b-a) x (c-a) if d-a is in the plane, but it's a tetrahedron.
                # Actually, the circumcenter of a tetrahedron with vertices on a sphere is the center of that sphere.
                # Since all points are on the unit sphere, the circumcenter is the origin (0,0,0).
                # But the Voronoi vertex is not the circumcenter of the simplex of points.
                # It is the point equidistant to the vertices of the facet.
                # For a facet (triangle) of the convex hull, the circumcenter of that triangle in 3D is the point on the normal line through the triangle's circumcenter that is equidistant to all 3 vertices.
                # Wait, the Voronoi vertex is equidistant to 4 points (the vertices of the tetrahedron formed by the point and its 3 neighbors? No).
                # Let's stick to the triangle case. In 3D, hull facets are triangles.
                # The Voronoi cell of point i is the intersection of half-spaces.
                # The vertices of the Voronoi cell of i are the circumcenters of the triangles in the hull that contain i?
                # No. The vertices of the Voronoi cell of i correspond to the facets of the convex hull that are "opposite" to i?
                # Actually, a standard algorithm:
                # 1. For each point i, find all triangles in the convex hull that have i as a vertex.
                # 2. For each such triangle (i, j, k), compute the circumcenter of the triangle (i, j, k).
                # 3. The spherical centroid is the normalized sum of these circumcenters.
                # This works if the Voronoi cell is defined by the perpendicular bisectors.
                # The circumcenter of (i, j, k) is equidistant to i, j, k.
                # It lies on the normal to the triangle.
                # This is a valid approximation for the spherical centroid.
                pass # Handled above
        if circumcenters:
            centroids[i] = _spherical_centroid(np.array(circumcenters))
        else:
            centroids[i] = points[i]
            
    return centroids

def _covering_radius(points):
    """Compute the covering radius in degrees."""
    # The covering radius is the maximum over all points on the sphere of the distance to the nearest point.
    # This is equivalent to the maximum circumradius of the facets of the convex hull.
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: use a grid
        # Create a grid on the sphere
        lat = np.linspace(-np.pi/2, np.pi/2, 100)
        lon = np.linspace(0, 2*np.pi, 100)
        LON, LAT = np.meshgrid(lon, lat)
        x = np.cos(LAT) * np.cos(LON)
        y = np.cos(LAT) * np.sin(LON)
        z = np.sin(LAT)
        grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        
        # Compute distances
        min_dists = np.inf
        for p in points:
            dists = np.arccos(np.clip(np.dot(grid, p), -1, 1))
            min_dists = np.minimum(min_dists, dists)
        return np.max(min_dists) * 180.0 / np.pi

    max_circumradius = 0
    for simplex in hull.simplices:
        pts = points[simplex]
        if len(pts) == 3:
            a, b, c = pts
            # Circumradius of triangle abc on the sphere
            # The circumcenter is the normalized cross product of (b-a) and (c-a)
            v1 = b - a
            v2 = c - a
            cross = np.cross(v1, v2)
            n = np.linalg.norm(cross)
            if n < 1e-12:
                continue
            circumcenter = cross / n
            # Distance from circumcenter to any vertex (e.g., a)
            dist = np.arccos(np.clip(np.dot(circumcenter, a), -1, 1))
            max_circumradius = max(max_circumradius, dist)
        elif len(pts) == 4:
            # Tetrahedron? In 3D, hull simplices are triangles.
            pass
            
    return max_circumradius * 180.0 / np.pi

def _local_optimize(points):
    """Local optimization using repulsion forces."""
    for _ in range(100):
        forces = np.zeros_like(points)
        for i in range(N):
            for j in range(N):
                if i == j: continue
                d = np.dot(points[i], points[j])
                if d > -0.99:
                    # Repulsive force
                    force = (1 - d) * (points[i] - d * points[j])
                    forces[i] += force
        # Normalize forces
        norms = np.linalg.norm(forces, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1
        forces /= norms
        # Update points
        points = points + 0.01 * forces
        # Normalize
        points = np.array([_normalize(p) for p in points])
    return points

def _simulated_annealing(points, time_limit):
    """Simulated annealing to escape local minima."""
    start_time = time.time()
    best_points = points.copy()
    best_radius = _covering_radius(best_points)
    current_points = points.copy()
    current_radius = best_radius
    T = 0.1
    cooling = 0.99
    
    while time.time() - start_time < time_limit:
        # Perturb
        idx = np.random.randint(N)
        noise = np.random.normal(0, 0.1, 3)
        new_point = _normalize(current_points[idx] + noise)
        new_points = current_points.copy()
        new_points[idx] = new_point
        
        new_radius = _covering_radius(new_points)
        
        if new_radius < current_radius or np.random.rand() < np.exp(-(new_radius - current_radius) / T):
            current_points = new_points
            current_radius = new_radius
            if current_radius < best_radius:
                best_points = current_points.copy()
                best_radius = current_radius
        
        T *= cooling
        if T < 1e-4:
            break
            
    return best_points, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 110s to leave margin for evaluation
    
    best_points = None
    best_radius = np.inf
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Local optimization
    points = _local_optimize(points)
    radius = _covering_radius(points)
    if radius < best_radius:
        best_radius = radius
        best_points = points.copy()
        
    # Spherical Lloyd's
    for _ in range(20):
        if time.time() > deadline:
            break
        centroids = _get_voronoi_centroids(points)
        # Blend to avoid instability
        points = np.array([_normalize(0.5 * p + 0.5 * c) for p, c in zip(points, centroids)])
        radius = _covering_radius(points)
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
            
    # Multi-restart simulated annealing
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        num_restarts = max(1, int(remaining_time / 10))
        for _ in range(num_restarts):
            if time.time() > deadline:
                break
            # Perturb best points
            perturbed = best_points.copy()
            for i in range(N):
                noise = np.random.normal(0, 0.5, 3)
                perturbed[i] = _normalize(perturbed[i] + noise)
            
            # Local optimize
            perturbed = _local_optimize(perturbed)
            
            # Simulated annealing
            sa_points, sa_radius = _simulated_annealing(perturbed, min(5, deadline - time.time()))
            if sa_radius < best_radius:
                best_radius = sa_radius
                best_points = sa_points.copy()
                
    return best_points
# EVOLVE-BLOCK-END
