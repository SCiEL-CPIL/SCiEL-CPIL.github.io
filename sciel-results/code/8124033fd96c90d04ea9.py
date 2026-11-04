import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

def _circumcenter(a, b, c):
    """
    Calculate the circumcenter of a spherical triangle defined by unit vectors a, b, c.
    Returns the unit vector circumcenter and the circumradius (angular distance).
    """
    # The circumcenter o satisfies o.a = o.b = o.c
    # o is in the plane spanned by a, b, c.
    # We can solve for o in the basis a, b, c.
    # Let o = x*a + y*b + z*c
    # o.a = x + y*(a.b) + z*(a.c)
    # o.b = x*(a.b) + y + z*(b.c)
    # o.c = x*(a.c) + y*(b.c) + z
    # We want o.a = o.b = o.c = cos(r)
    
    # Matrix A such that A * [x,y,z]^T = [1,1,1]^T * cos(r) ? No.
    # Let's set o.a = o.b.
    # x(1 - a.b) + y(a.b - 1) + z(a.c - b.c) = 0
    # x(a.c - a.b) + y(b.c - 1) + z(1 - b.c) = 0  (from o.b = o.c)
    
    # Actually, easier:
    # The circumcenter is the normalized vector:
    # o = (a * sin(2*angle_b) + b * sin(2*angle_c) + c * sin(2*angle_a))
    # where angle_a is the angle at vertex a, etc.
    # This formula is for the circumcenter of the spherical triangle.
    
    # Let's use the linear algebra approach which is robust.
    # We want o such that o is orthogonal to (a-b) and (b-c)? No.
    # o is equidistant from a, b, c.
    # ||o||^2 - 2 o.a + ||a||^2 = ||o||^2 - 2 o.b + ||b||^2
    # 2 o.(b-a) = ||b||^2 - ||a||^2 = 0 (since unit vectors)
    # So o is orthogonal to (b-a) and (c-a).
    # Therefore o is parallel to (b-a) x (c-a).
    
    n = np.cross(b - a, c - a)
    norm_n = np.linalg.norm(n)
    if norm_n < 1e-12:
        return a, 0.0 # Degenerate
    
    o = n / norm_n
    
    # Check orientation. The circumcenter should be "inside" or on the correct side.
    # For a triangle on the sphere, the circumcenter can be the "antipodal" one if the triangle is large.
    # We want the center of the cap.
    # The angular distance r is arccos(o.a).
    # If the triangle is acute, o.a > 0. If obtuse, o.a < 0?
    # Actually, if the triangle contains the pole, the circumcenter might be on the other side.
    # We want the circumcenter that is closest to the centroid?
    # Let's check the dot product with the centroid.
    centroid = (a + b + c) / 3.0
    centroid = centroid / np.linalg.norm(centroid)
    
    if np.dot(o, centroid) < 0:
        o = -o
        
    r = np.arccos(np.clip(np.dot(o, a), -1.0, 1.0))
    return o, r

def _largest_empty_circle_center(a, b, c, points):
    """
    Given a Delaunay triangle (a,b,c) and all points, determine the center of the largest empty circle
    associated with this triangle's Voronoi vertex.
    If the circumcenter is valid (closer to a,b,c than any other point), return it.
    Otherwise, the largest empty circle center is the midpoint of the longest edge.
    """
    o, r_circ = _circumcenter(a, b, c)
    
    # Check if o is closer to a, b, c than to any other point
    # dist(o, a) = r_circ
    # dist(o, p) for p in points
    # We only need to check if o is in the Voronoi cell of a (or b or c).
    # If o is in the Voronoi cell of a, then o is closer to a than to any other point.
    # Since o is equidistant to a,b,c, if it's closer to a than to d, it's closer to b and c than to d.
    
    dists_to_o = np.arccos(np.clip(points @ o, -1.0, 1.0))
    min_dist_to_o = np.min(dists_to_o)
    
    # If the closest point to o is one of a,b,c (dist approx r_circ), then o is a valid Voronoi vertex.
    # Due to floating point, we check if min_dist_to_o is close to r_circ.
    if abs(min_dist_to_o - r_circ) < 1e-8:
        return o, r_circ
    else:
        # The circumcenter is outside the Voronoi cell.
        # The largest empty circle is centered at the midpoint of the longest edge.
        # Calculate edge lengths (angular)
        d_ab = np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))
        d_bc = np.arccos(np.clip(np.dot(b, c), -1.0, 1.0))
        d_ca = np.arccos(np.clip(np.dot(c, a), -1.0, 1.0))
        
        if d_ab >= d_bc and d_ab >= d_ca:
            mid = (a + b) / 2.0
            mid = mid / np.linalg.norm(mid)
            r_mid = d_ab / 2.0
            return mid, r_mid
        elif d_bc >= d_ab and d_bc >= d_ca:
            mid = (b + c) / 2.0
            mid = mid / np.linalg.norm(mid)
            r_mid = d_bc / 2.0
            return mid, r_mid
        else:
            mid = (c + a) / 2.0
            mid = mid / np.linalg.norm(mid)
            r_mid = d_ca / 2.0
            return mid, r_mid

def construct():
    start_time = time.time()
    deadline = 110.0 # 120s limit, keep 10s margin
    
    # Initial Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = float('inf')
    
    # Precompute indices for hull facets
    # We will iterate and update points
    
    step_size = 0.1
    cooling = 0.99
    
    for iter in range(1000):
        if time.time() - start_time > deadline:
            break
            
        try:
            hull = ConvexHull(points)
            facets = hull.simplices
        except Exception:
            # If hull fails (e.g., coplanar points), jitter slightly
            points += np.random.normal(0, 1e-5, points.shape)
            continue
            
        # For each point, accumulate a move vector
        moves = np.zeros_like(points)
        
        # Map point index to list of facets it belongs to
        point_facets = [[] for _ in range(N)]
        for idx, facet in enumerate(facets):
            for p_idx in facet:
                point_facets[p_idx].append(idx)
        
        # Calculate circumcenters and radii for all facets
        facet_centers = np.zeros((len(facets), 3))
        facet_radii = np.zeros(len(facets))
        
        for idx, facet in enumerate(facets):
            a, b, c = points[facet[0]], points[facet[1]], points[facet[2]]
            center, radius = _largest_empty_circle_center(a, b, c, points)
            facet_centers[idx] = center
            facet_radii[idx] = radius
            
        # Update points
        for i in range(N):
            if not point_facets[i]:
                continue
                
            # Find the "worst" facet for this point (largest radius)
            # Or average the influence?
            # Lloyd's algorithm: move to the center of the largest empty circle in the Voronoi cell.
            # The largest empty circle in the Voronoi cell of i is the one with the largest radius
            # among the Voronoi vertices (circumcenters/midpoints) associated with i.
            
            max_r = -1
            best_center = points[i]
            
            for f_idx in point_facets[i]:
                r_f = facet_radii[f_idx]
                if r_f > max_r:
                    max_r = r_f
                    best_center = facet_centers[f_idx]
                    
            # Move towards best_center
            # Step size should be proportional to the distance?
            # Or fixed step?
            # Let's use a step that is a fraction of the distance to the center.
            diff = best_center - points[i]
            dist = np.linalg.norm(diff)
            if dist > 1e-12:
                # Move a fraction of the way
                # If we move all the way, it might oscillate.
                # step = 0.5 * dist?
                # Let's use a fixed step size scaled by step_size
                step = step_size * (diff / dist)
                moves[i] += step
                
        # Apply moves
        points += moves
        
        # Normalize to sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
        # Calculate current covering radius (max facet radius)
        current_radius = np.max(facet_radii)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
            
        # Cool down
        step_size *= cooling
        
    return best_points

construct()
