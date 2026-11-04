"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _covering_radius(points):
    """Compute the covering radius in degrees for points on the unit sphere."""
    # The covering radius is the maximum over all points y on the sphere of
    # the angular distance to the nearest chosen point.
    # This equals the maximum circumradius of the facets of the convex hull.
    
    # Check if points are in a closed hemisphere (invalid)
    # A set of points is in a closed hemisphere if there exists a vector v
    # such that v . p_i >= 0 for all i.
    # We check this by trying to find a separating hyperplane.
    
    # First, normalize points
    norms = np.linalg.norm(points, axis=1)
    if np.any(norms < 1e-12):
        return 180.0  # Invalid
    points = points / norms[:, None]
    
    # Check for coincident points
    for i in range(N):
        for j in range(i+1, N):
            if np.linalg.norm(points[i] - points[j]) < 1e-10:
                return 180.0  # Invalid
    
    # Check if all points are in a closed hemisphere
    # Try to find a vector v such that v . p_i >= 0 for all i
    # This is equivalent to checking if the origin is in the convex hull
    # Actually, points are in a closed hemisphere iff the origin is NOT in
    # the interior of their convex hull. Let's check if there's a separating plane.
    
    # Simple check: try random directions and see if all points project positively
    # More robust: check if the convex hull contains the origin
    try:
        hull = ConvexHull(points)
    except Exception:
        return 180.0
    
    # Check if origin is inside the convex hull
    # The origin is inside if for every facet, the origin is on the same side
    # as the interior (or on the facet)
    # For a convex hull in 3D, we can check by looking at the normals
    # Actually, let's use a different approach:
    # Points are in a closed hemisphere iff there exists a unit vector v
    # such that v . p_i >= 0 for all i.
    
    # We can check this by solving: minimize max_i (v . p_i) over ||v||=1
    # If the minimum is >= 0, then they're in a closed hemisphere.
    # Actually, we want to check if max_i (v . p_i) >= 0 for some v.
    # Equivalently, min_v max_i (v . p_i) >= 0 means NOT in a hemisphere?
    # No. Let me think again.
    
    # Points p_1, ..., p_N are in a closed hemisphere iff there exists a unit vector v
    # such that v . p_i >= 0 for all i.
    # This is equivalent to: the convex hull of {p_1, ..., p_N} does not contain the origin in its interior.
    # Or: the origin is not in the interior of conv{p_i}.
    
    # Let's check if origin is in the convex hull interior using the hull
    # For each facet, the inward normal should point away from origin if origin is inside
    # Actually, simpler: check if there's a vector v with v.p_i > 0 for all i (strict)
    # or v.p_i >= 0 for all i (non-strict).
    
    # Let's just compute the covering radius and also validate
    # For the covering radius, we look at each facet of the convex hull.
    # For a facet with vertices v_1, v_2, ..., v_k, the circumcenter of the facet
    # (projected to the sphere) gives the farthest point in that facet's cap.
    
    # Actually, the covering radius is the maximum over all facets of the angular
    # distance from the facet's circumcenter (on the sphere) to any vertex of the facet.
    # But only if the circumcenter is within the facet. If not, the farthest point
    # is at a vertex or edge.
    
    # More precisely: for each facet, compute the circumcenter of the spherical polygon.
    # The covering radius is the max over facets of the max angular distance from
    # the circumcenter to the vertices of that facet, but only if the circumcenter
    # lies within the facet. Otherwise, we need to check the boundary.
    
    # A simpler approach: the covering radius equals the maximum, over all points
    # on the sphere, of the distance to the nearest point. This maximum occurs at
    # either a vertex of the Voronoi diagram (circumcenter of a Delaunay triangle)
    # or on an edge, or at a vertex.
    
    # For points on a sphere, the covering radius is the maximum circumradius of
    # the Delaunay triangles (or more generally, the facets of the convex hull).
    
    # Let's compute it properly:
    # For each facet of the convex hull, find the point on the sphere that is
    # farthest from all vertices of that facet but closest to the facet's interior.
    # This is the circumcenter of the facet if it lies within the facet, otherwise
    # it's on the boundary.
    
    # Actually, the standard result is: the covering radius is the maximum, over all
    # facets F of the convex hull, of the angular circumradius of F, provided the
    # circumcenter of F lies within F. If not, the farthest point in that region
    # is on the boundary of F, which is covered by adjacent facets.
    
    # So we just need to check each facet: compute the circumcenter, check if it's
    # within the facet, and if so, compute the angular distance to the vertices.
    
    max_rad = 0.0
    for simplex in hull.simplices:
        verts = points[simplex]
        # Compute circumcenter of the triangle formed by these 3 points
        # The circumcenter in 3D of a triangle on the unit sphere
        # is the unit vector in the direction of the cross product normal,
        # but actually it's the point on the sphere equidistant from all 3 vertices.
        
        # For 3 points on a sphere, the circumcenter is the intersection of the
        # sphere with the plane perpendicular to the plane of the triangle, passing
        # through the centroid... no, that's not right.
        
        # The circumcenter of 3 points on a sphere is the point on the sphere
        # equidistant from all 3. This is found by:
        # Let a, b, c be the 3 points.
        # The circumcenter x satisfies |x-a| = |x-b| = |x-c| and |x| = 1.
        # |x-a|^2 = |x-b|^2 => -2x.a + |a|^2 = -2x.b + |b|^2 => 2x.(b-a) = |b|^2 - |a|^2
        # Since |a|=|b|=|c|=1, we get x.(b-a) = 0 and x.(c-a) = 0.
        # So x is perpendicular to (b-a) and (c-a), meaning x is parallel to (b-a) x (c-a).
        
        cross = np.cross(verts[1] - verts[0], verts[2] - verts[0])
        norm_cross = np.linalg.norm(cross)
        if norm_cross < 1e-12:
            continue
        circumcenter = cross / norm_cross
        
        # Check which side: the circumcenter could be on either side
        # We need the one that is "outside" the hull, i.e., the one that maximizes
        # distance to the vertices. Actually, both are equidistant, but we want
        # the one on the sphere that is in the cap.
        
        # The correct circumcenter is the one such that the angular distance to
        # each vertex is less than 90 degrees (for a proper cap).
        # Actually, let's compute the angular distance and pick the sign that
        # gives the smaller distances.
        
        d1 = np.arccos(np.clip(np.dot(circumcenter, verts[0]), -1, 1))
        d2 = np.arccos(np.clip(np.dot(-circumcenter, verts[0]), -1, 1))
        
        if d1 < d2:
            cc = circumcenter
        else:
            cc = -circumcenter
        
        # Check if cc is within the facet (i.e., the cap defined by the facet
        # contains cc). This means cc should be on the same side of the facet
        # as the hull interior... actually, for the covering radius, we need
        # cc to be in the region of the sphere covered by this facet's Voronoi cell.
        
        # A simpler check: the angular distance from cc to each vertex should be
        # the same, and cc should be "visible" from the facet.
        
        # Check if cc is within the spherical triangle:
        # The barycentric coordinates should all be non-negative.
        # For a spherical triangle, we can check by verifying that cc is on the
        # same side of each edge as the opposite vertex.
        
        # Edge 0-1: check if cc and vertex 2 are on the same side
        # The plane through 0, 1, and origin has normal n01 = verts[0] x verts[1]
        n01 = np.cross(verts[0], verts[1])
        n01_norm = np.linalg.norm(n01)
        if n01_norm < 1e-12:
            continue
        n01 = n01 / n01_norm
        
        side_cc = np.dot(n01, cc)
        side_v2 = np.dot(n01, verts[2])
        
        # The facet is the set of points on the sphere that are on the same side
        # as the hull interior. For the convex hull, the interior is on the side
        # where the normal points inward. Let's determine the correct normal direction.
        
        # The outward normal of the facet points away from the hull center.
        # The hull center is the origin (since points are on the unit sphere).
        # So the outward normal should have negative dot product with the centroid.
        centroid = np.mean(verts, axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm < 1e-12:
            continue
        centroid = centroid / centroid_norm
        
        # Outward normal: the one with negative dot with centroid
        if np.dot(n01, centroid) > 0:
            n01 = -n01
        
        # Now n01 is the outward normal. Points on the facet have n01 . p = 0
        # (approximately). The interior of the hull has n01 . p < 0.
        # The cap (exterior) has n01 . p > 0.
        
        # cc should be in the cap: n01 . cc > 0
        # And the opposite vertex should be in the hull: n01 . verts[2] < 0
        # Wait, this is for the planar triangle. For the spherical triangle,
        # the condition is different.
        
        # Let's use a different approach: check if cc is within the spherical
        # triangle by verifying that the angular distances satisfy certain conditions.
        
        # Actually, for the covering radius computation, we can just compute
        # the circumradius and check if the circumcenter is valid by ensuring
        # that the angular distances to all 3 vertices are equal and less than
        # the distances to other points.
        
        # Let's just compute the circumradius for each facet and take the max.
        # The covering radius is the max circumradius over all facets where
        # the circumcenter lies within the facet.
        
        # Check if cc is within the spherical triangle using the sign test:
        # For each edge, cc and the opposite vertex should be on opposite sides
        # of the great circle through the edge (if cc is inside) or same side...
        
        # I'll use a more robust check: compute the solid angle or use
        # barycentric coordinates on the sphere.
        
        # For now, let's just compute the circumradius and assume it's valid
        # if the angular distances are reasonable.
        
        ang_dist = np.arccos(np.clip(np.dot(cc, verts[0]), -1, 1))
        
        # Check if cc is actually the farthest point in this facet's Voronoi cell.
        # This is true if cc is within the spherical triangle.
        # Simple check: the angular distance from cc to each edge midpoint should
        # be less than the distance to the vertices... this is getting complicated.
        
        # Let's just use the circumradius and filter by checking if the
        # circumcenter is on the correct side of the facet.
        
        # The facet's outward normal is:
        facet_normal = np.cross(verts[1] - verts[0], verts[2] - verts[0])
        fn_norm = np.linalg.norm(facet_normal)
        if fn_norm < 1e-12:
            continue
        facet_normal = facet_normal / fn_norm
        
        # Make sure it points outward (away from origin)
        if np.dot(facet_normal, centroid) > 0:
            facet_normal = -facet_normal
        
        # cc should be on the outside: facet_normal . cc > 0
        if np.dot(facet_normal, cc) < 0:
            cc = -cc
            if np.dot(facet_normal, cc) < 0:
                continue  # Invalid circumcenter
        
        # Now compute the circumradius
        circumradius = np.arccos(np.clip(np.dot(cc, verts[0]), -1, 1))
        
        # Check if cc is within the spherical triangle by verifying that
        # for each edge, cc and the opposite vertex are on the same side
        # of the great circle (for cc to be inside).
        
        valid = True
        for i in range(3):
            v1 = verts[i]
            v2 = verts[(i+1) % 3]
            v3 = verts[(i+2) % 3]
            
            # Great circle through v1 and v2: normal is v1 x v2
            gc_normal = np.cross(v1, v2)
            gc_norm = np.linalg.norm(gc_normal)
            if gc_norm < 1e-12:
                continue
            gc_normal = gc_normal / gc_norm
            
            side_cc = np.dot(gc_normal, cc)
            side_v3 = np.dot(gc_normal, v3)
            
            # For cc to be inside the spherical triangle, cc and v3 should be
            # on the same side of the great circle through v1 and v2.
            # But we need to be careful about the orientation.
            if side_cc * side_v3 < -1e-12:
                valid = False
                break
        
        if valid:
            max_rad = max(max_rad, circumradius)
    
    return np.degrees(max_rad)


def _optimize_covering(points, time_budget=100.0):
    """Optimize the covering radius using local search."""
    start_time = time.time()
    points = points.copy()
    current_rad = _covering_radius(points)
    
    # Gradient-free local optimization
    # Try moving each point in random directions
    
    best_points = points.copy()
    best_rad = current_rad
    
    step_size = 0.1
    n_iter = 0
    
    while time.time() - start_time < time_budget:
        n_iter += 1
        improved = False
        
        # Try moving each point
        for i in range(N):
            # Generate random direction
            direction = np.random.randn(3)
            direction /= np.linalg.norm(direction)
            
            # Try small steps
            for sign in [1, -1]:
                new_points = points.copy()
                new_points[i] = points[i] + sign * step_size * direction
                new_points[i] /= np.linalg.norm(new_points[i])
                
                new_rad = _covering_radius(new_points)
                if new_rad < best_rad - 1e-10:
                    best_rad = new_rad
                    best_points = new_points
                    improved = True
                    break
        
        if not improved:
            # Reduce step size
            step_size *= 0.9
            if step_size < 1e-6:
                break
        else:
            points = best_points.copy()
    
    return best_points, best_rad


def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s safety margin
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    # Local optimization
    points, rad = _optimize_covering(points, time_budget=min(100.0, deadline - (time.time() - start_time)))
    
    # Check time
    if time.time() - start_time > deadline:
        # Return best we have
        pass
    
    # Final normalization
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    return points
# EVOLVE-BLOCK-END
