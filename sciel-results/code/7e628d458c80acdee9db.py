"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

def _normalize(v):
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms

def _covering_radius(points):
    """
    Compute the covering radius by finding the circumradius of the largest facet
    of the convex hull. The covering radius is the maximum over all points on the
    sphere of the angular distance to the nearest point. This is equivalent to the
    maximum circumradius of the facets of the convex hull of the points (assuming
    the origin is inside the hull, which it is for a good covering).
    """
    try:
        hull = ConvexHull(points)
        # For each facet, compute the circumradius
        # The circumradius of a spherical triangle (facet) can be computed from its side lengths
        # or by finding the center of the circumscribed circle on the sphere.
        
        # A more robust way: The covering radius is the maximum distance from the origin to the 
        # plane of the facet? No.
        # The distance from the origin to the plane of the facet is the inradius of the dual polytope?
        
        # Let's use the property: The covering radius is the maximum over all facets of the 
        # circumradius of that facet.
        # For a facet defined by vertices v1, v2, ..., vk, the circumcenter on the sphere is the 
        # point c that minimizes max_i angle(c, vi).
        
        # Alternative: The distance from the origin to the plane of the facet is h.
        # The circumradius R of the facet (on the unit sphere) satisfies:
        # The plane is n . x = h, where n is the unit normal.
        # The intersection of the plane with the sphere is a circle of radius r = sqrt(1 - h^2).
        # The circumradius of the facet (the distance from the circumcenter to the vertices) is r.
        # Wait, the circumcenter of the facet is the point on the sphere closest to the origin 
        # along the normal? No.
        
        # Let's use the fact that the covering radius is the maximum over all points y on the sphere 
        # of min_i angle(y, p_i).
        # This is the same as the maximum circumradius of the Voronoi cells.
        
        # Efficient method:
        # 1. Compute the convex hull.
        # 2. For each facet, compute the distance from the origin to the plane.
        # 3. The circumradius of the facet (on the sphere) is arcsin(sqrt(1 - d^2))? No.
        
        # Let's use the definition: The covering radius is the maximum over all facets of the 
        # distance from the circumcenter of the facet to its vertices.
        
        # For a triangle on the sphere with sides a, b, c, the circumradius R is given by:
        # sin(R) = abc / (4 * Area)
        # where Area is the spherical area.
        
        # This is complex. Let's use a simpler approach:
        # The covering radius is the maximum over all facets of the circumradius.
        # We can compute the circumradius of each facet by finding the center of the 
        # circumscribed circle.
        
        # For a facet with vertices v1, v2, ..., vk, the circumcenter c is the solution to:
        # |c - vi|^2 = R^2 for all i.
        # This implies c . vi = c . vj for all i, j.
        # So c is orthogonal to vi - vj for all i, j.
        # Thus c is parallel to the cross product of (v1 - v2) and (v1 - v3).
        
        # Let's compute the normal to the facet.
        # The plane of the facet has normal n. The distance from the origin to the plane is h = n . v1.
        # The circle of intersection has radius r = sqrt(1 - h^2).
        # The circumradius of the facet is the distance from the center of this circle to the vertices.
        # The center of the circle is c = n / h (if h > 0).
        # Wait, the center of the circle of intersection is the projection of the origin onto the plane, 
        # which is n * h. No, the plane is n . x = h. The projection of the origin is n * h.
        # The distance from n*h to v1 is sqrt(|v1|^2 - h^2) = sqrt(1 - h^2).
        # So the circumradius of the facet is sqrt(1 - h^2)? No, that's the radius of the circle, 
        # but the circumradius of the polygon inscribed in the circle is not necessarily the radius 
        # of the circle.
        
        # Actually, for a spherical polygon, the circumradius is the distance from the circumcenter 
        # to the vertices. The circumcenter is the point on the sphere that is equidistant from all vertices.
        
        # Let's use a different approach: The covering radius is the maximum over all points y on the 
        # sphere of min_i angle(y, p_i). We can approximate this by checking the circumcenters of the 
        # facets of the convex hull.
        
        # For each facet, the circumcenter is the point on the sphere that is equidistant from the 
        # vertices of the facet. This point is the one that maximizes the minimum distance to the 
        # vertices within that facet's Voronoi cell.
        
        # Let's compute the circumcenter for each facet.
        # For a triangle with vertices a, b, c, the circumcenter is the solution to:
        # x . a = x . b = x . c
        # x . x = 1
        # This is a linear system.
        
        # Let's use the fact that the circumcenter is parallel to the cross product of the 
        # differences of the vertices.
        
        # For a triangle with vertices a, b, c:
        # u = b - a, v = c - a
        # n = u x v
        # The circumcenter is in the direction of n? No.
        
        # Let's use the formula for the circumradius of a spherical triangle:
        # sin(R) = sin(a)sin(b)sin(c) / (2 * sin(s)) where s is the semiperimeter? No.
        
        # Let's use a simpler method: 
        # The covering radius is the maximum over all facets of the circumradius.
        # We can compute the circumradius of each facet by finding the center of the 
        # circumscribed circle.
        
        # For a facet with vertices v1, ..., vk, the circumcenter c satisfies:
        # c . vi = c . vj for all i, j.
        # This means c is orthogonal to vi - vj for all i, j.
        # So c is in the null space of the matrix with rows vi - vj.
        # For a triangle, c is parallel to (v1 - v2) x (v1 - v3).
        
        # Let's compute the circumcenter for each facet.
        facets = hull.simplices
        max_r = 0.0
        for simplex in facets:
            verts = points[simplex]
            # Compute the circumcenter
            # For a triangle:
            if len(verts) == 3:
                a, b, c = verts
                # Solve for x such that x . a = x . b = x . c and x . x = 1
                # x . (a - b) = 0
                # x . (a - c) = 0
                # x . x = 1
                # This means x is parallel to (a - b) x (a - c)
                n = np.cross(a - b, a - c)
                n_norm = np.linalg.norm(n)
                if n_norm < 1e-10:
                    continue
                n = n / n_norm
                # The circumcenter is either n or -n. Choose the one that is closer to the centroid.
                centroid = (a + b + c) / 3.0
                centroid_norm = np.linalg.norm(centroid)
                if centroid_norm < 1e-10:
                    continue
                centroid = centroid / centroid_norm
                if np.dot(n, centroid) < 0:
                    n = -n
                # The distance from n to a is the circumradius
                d = np.arccos(np.clip(np.dot(n, a), -1.0, 1.0))
                if d > max_r:
                    max_r = d
            else:
                # For higher-dimensional facets, we can use the same idea
                # But the hull of points on the sphere in 3D will have triangular facets
                pass
        return max_r * 180.0 / np.pi
    except:
        return np.inf

def _lloyd_relax(points, steps=10):
    """
    Perform Lloyd's relaxation: move each point to the centroid of its Voronoi cell.
    """
    points = _normalize(points)
    for _ in range(steps):
        try:
            hull = ConvexHull(points)
            # For each point, find the centroid of its Voronoi cell
            # The Voronoi cell of point i is the intersection of the half-spaces 
            # defined by the perpendicular bisectors of the edges from i to its neighbors.
            # A simpler approach: For each point, find the centroid of the facets 
            # that contain it.
            new_points = np.zeros_like(points)
            counts = np.zeros(N)
            for simplex in hull.simplices:
                centroid = np.mean(points[simplex], axis=0)
                centroid_norm = np.linalg.norm(centroid)
                if centroid_norm < 1e-10:
                    continue
                centroid = centroid / centroid_norm
                for idx in simplex:
                    new_points[idx] += centroid
                    counts[idx] += 1
            # Update points
            for i in range(N):
                if counts[i] > 0:
                    new_points[i] = new_points[i] / counts[i]
            points = _normalize(new_points)
        except:
            break
    return points

def _random_perturbation(points, sigma=0.1):
    """
    Apply a random perturbation to the points.
    """
    # Generate random directions
    random_dirs = np.random.randn(N, 3)
    random_dirs = _normalize(random_dirs)
    # Perturb
    perturbed = points + sigma * random_dirs
    return _normalize(perturbed)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140 seconds to be safe
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points = _normalize(points)
    
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    # Multi-start basin hopping
    num_restarts = 20
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Random perturbation
        current_points = _random_perturbation(points, sigma=0.5)
        current_points = _lloyd_relax(current_points, steps=20)
        
        current_radius = _covering_radius(current_points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = current_points.copy()
            points = current_points.copy()
        else:
            # Accept with some probability (simulated annealing-like)
            # But for simplicity, we can just accept if it's not too bad
            if current_radius < best_radius * 1.1:
                points = current_points.copy()
    
    return best_points
# EVOLVE-BLOCK-END
