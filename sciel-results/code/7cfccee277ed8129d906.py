import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.spatial import cKDTree
import sys

N = 130
DEADLINE = 110.0  # Leave 40s margin for safety/evaluation overhead? No, 150s kill. 110s is safe.

def _fibonacci_sphere(n, offset=0.5):
    """Generate n points on a unit sphere using the Fibonacci spiral."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(1.0 - z * z)
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.column_stack([x, y, z])

def _normalize(points):
    """Normalize points to the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return points / norms

def _covering_radius(points):
    """
    Compute the exact covering radius in radians.
    The covering radius is the maximum angular distance from any point on the sphere
    to the nearest point in the set. This is equivalent to the maximum circumradius
    of the facets of the convex hull of the points (projected to the sphere).
    """
    if len(points) < 4:
        return np.pi / 2.0
        
    try:
        hull = ConvexHull(points)
    except Exception:
        # If hull fails, points might be coplanar or degenerate.
        # Fallback to a grid-based estimation or return a large value.
        # For robustness, return pi/2 (hemisphere) as a penalty.
        return np.pi / 2.0

    max_radius = 0.0
    # hull.simplices gives the indices of the vertices for each facet
    for simplex in hull.simplices:
        p1 = points[simplex[0]]
        p2 = points[simplex[1]]
        p3 = points[simplex[2]]
        
        # Calculate circumcenter of the triangle p1, p2, p3 on the unit sphere.
        # The circumcenter is the intersection of the sphere with the line passing through
        # the circumcenter of the triangle in 3D.
        # Alternatively, solve for a vector c such that |c|^2=1 and c.p1 = c.p2 = c.p3.
        # This means c is orthogonal to (p1-p2) and (p1-p3).
        # So c is parallel to the cross product (p1-p2) x (p1-p3).
        
        v1 = p1 - p2
        v2 = p1 - p3
        normal = np.cross(v1, v2)
        
        norm_normal = np.linalg.norm(normal)
        if norm_normal < 1e-12:
            # Degenerate triangle (collinear points on sphere? impossible for distinct points unless antipodal issues)
            # If points are antipodal, the "triangle" is degenerate.
            # In this case, the covering radius is likely determined by the antipodal pair.
            # Distance to midpoint is 0, distance to poles is pi/2.
            # But for a facet, if it's degenerate, we should check the edges.
            # For now, assume non-degenerate.
            continue
            
        c = normal / norm_normal
        # Ensure c is on the same side as the origin? 
        # The circumcenter of a spherical triangle is the point equidistant to vertices.
        # The normal vector points to the circumcenter. We need to pick the sign such that
        # the angle is acute/obtuse correctly? 
        # Actually, the circumcenter is unique. The vector c calculated is the normal to the plane.
        # The circumcenter is at distance R from the origin? No, it's ON the sphere.
        # Wait, the circumcenter of the *spherical* triangle is the point on the sphere.
        # The 3D circumcenter of the triangle p1,p2,p3 is at distance h from origin.
        # The point on the sphere is c = normal / |normal| ? No.
        # Let's verify.
        # If p1,p2,p3 are on unit sphere, the plane they define is at distance d from origin.
        # The normal is n. The circumcenter in the plane is p_c = d*n.
        # The point on the sphere is n (or -n).
        # We need to choose the sign. The circumcenter of the spherical triangle is the one 
        # that is "inside" the triangle? Or rather, the one that minimizes the max distance?
        # Actually, for a convex hull facet, the circumcenter is the vertex of the Voronoi diagram.
        # It is the point on the sphere equidistant to p1, p2, p3.
        # There are two such points (antipodal). One is the circumcenter of the triangle,
        # the other is the circumcenter of the "antipodal" triangle.
        # The one that is part of the Voronoi diagram for the *set* of points is the one
        # that is closer to the points? No, it's the one that is a local maximum of the distance function?
        # Actually, the Voronoi vertex is the point that is equidistant to the nearest points.
        # For a convex hull facet, the circumcenter is the one that is "outside" the sphere relative to the facet?
        # Let's just calculate the angular distance to p1 for both c and -c.
        # The correct circumcenter for the facet of the convex hull is the one that 
        # forms an acute triangle with the origin? Or simply, we take the one that is 
        # consistent with the hull orientation.
        
        # Simpler check: The covering radius is the max angular distance.
        # For a facet, the "worst" point is the circumcenter.
        # We calculate the angle between c and p1.
        # If we pick the wrong sign, the angle might be > pi/2.
        # The circumradius of a spherical triangle is <= pi/2 if the triangle is contained in a hemisphere.
        # Since these are facets of a convex hull of points on the sphere, the triangle is "small" 
        # (area < 2pi) and contained in a hemisphere? 
        # Yes, facets of the convex hull of points on the sphere correspond to Delaunay triangles 
        # which are contained in a hemisphere (unless the points cover the whole sphere in a way that 
        # a triangle is > hemisphere, which implies the origin is inside the tetrahedron? No).
        # Actually, if the points are not contained in a hemisphere, the convex hull contains the origin.
        # In that case, the "facets" are still triangles, but their circumcenters might be tricky.
        # However, the covering radius is still the max distance to the nearest point.
        # The Voronoi diagram on the sphere is still defined.
        
        # Let's just compute the angle for c and -c and take the smaller one?
        # No, the circumcenter is unique.
        # Let's use the property that c . p1 = c . p2 = c . p3 = cos(theta).
        # theta is the circumradius.
        # We want the theta such that the point c is the vertex of the Voronoi cell.
        # For a convex hull, the circumcenter is the one where the origin is on the "same side" 
        # of the facet as the circumcenter? No.
        
        # Let's just compute both and pick the one that is closer to the points?
        # Actually, if the triangle is small, both c and -c are far. 
        # Wait, if p1, p2, p3 are close, c is close to them. -c is far.
        # So we should pick the c that is closer to p1.
        
        dot1 = np.dot(c, p1)
        dot_neg = np.dot(-c, p1)
        
        # The circumradius is arccos(max(dot1, dot_neg))? 
        # No, the circumradius is the angle. 
        # If we pick c such that dot(c, p1) is maximized (closest), then the angle is arccos(dot).
        # Is it always the case that the Voronoi vertex is the one closer to the vertices?
        # Yes, the Voronoi cell is the set of points closer to p_i than any other.
        # The vertex is the center of the largest empty circle (on the sphere).
        # So we want the point c on the sphere that is equidistant to p1,p2,p3 and 
        # has no other points closer to it.
        # The "closest" circumcenter is the correct one for the local Voronoi vertex.
        
        if dot1 > dot_neg:
            cos_theta = dot1
        else:
            cos_theta = dot_neg
            
        # Clamp for numerical stability
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        
        if theta > max_radius:
            max_radius = theta
            
    return max_radius

def _lloyd_step(points, k=8):
    """
    Perform one step of Lloyd's algorithm (Centroidal Voronoi).
    Move each point to the centroid of its k nearest neighbors.
    """
    tree = cKDTree(points)
    # Query k+1 nearest (including self)
    dists, inds = tree.query(points, k=k+1)
    
    new_points = np.zeros_like(points)
    for i in range(N):
        neighbors = points[inds[i, 1:]] # Exclude self
        # Average the neighbors
        centroid = np.mean(neighbors, axis=0)
        new_points[i] = centroid
        
    return _normalize(new_points)

def _local_search(points, steps=50, step_size=0.05):
    """
    Simple random local search to fine-tune the covering radius.
    """
    current_points = points.copy()
    current_radius = _covering_radius(current_points)
    best_points = current_points.copy()
    best_radius = current_radius
    
    np.random.seed(42)
    
    for _ in range(steps):
        # Pick a random point to perturb
        idx = np.random.randint(N)
        # Generate a random direction
        direction = np.random.normal(size=3)
        direction = direction / np.linalg.norm(direction)
        
        # Perturb
        new_point = current_points[idx] + step_size * direction
        new_point = new_point / np.linalg.norm(new_point)
        
        candidate_points = current_points.copy()
        candidate_points[idx] = new_point
        
        candidate_radius = _covering_radius(candidate_points)
        
        if candidate_radius < current_radius:
            current_points = candidate_points
            current_radius = candidate_radius
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = current_points.copy()
        else:
            # Revert
            pass
            
    return best_points, best_radius

def construct():
    start_time = time.time()
    
    # Initial configuration
    x0 = _fibonacci_sphere(N, offset=0.5)
    best_points = x0.copy()
    best_radius = _covering_radius(best_points)
    
    # Phase 1: Lloyd's Algorithm
    # Lloyd's is fast and robust. We run it until time runs out or convergence.
    while time.time() - start_time < DEADLINE - 10:
        new_points = _lloyd_step(best_points, k=10)
        new_radius = _covering_radius(new_points)
        
        if new_radius < best_radius:
            best_radius = new_radius
            best_points = new_points
        else:
            # If it didn't improve, maybe try a different k or break?
            # Lloyd's usually improves or plateaus. 
            # To avoid getting stuck in a local minimum of the Lloyd iteration,
            # we can occasionally add a small random perturbation.
            if np.random.random() < 0.05:
                noise = np.random.normal(0, 0.01, size=(N, 3))
                new_points = _normalize(best_points + noise)
                new_radius = _covering_radius(new_points)
                if new_radius < best_radius:
                    best_radius = new_radius
                    best_points = new_points

    # Phase 2: Local Search
    # Use remaining time for fine-tuning
    time_left = DEADLINE - (time.time() - start_time)
    if time_left > 5:
        # Estimate how many steps we can do
        # _covering_radius is O(N^2) or so. 
        # Let's do a fixed number of steps, say 100, or until time runs out.
        current_points, current_radius = _local_search(best_points, steps=100, step_size=0.02)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = current_points
            
    return _normalize(best_points)

if __name__ == "__main__":
    # Quick test
    pts = construct()
    print(f"Covering Radius: {np.degrees(_covering_radius(pts))} degrees")
