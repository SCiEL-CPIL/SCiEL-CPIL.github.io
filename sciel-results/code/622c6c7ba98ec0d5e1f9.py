"""
Strategy: Iterative Repulsion with Delaunay Facet Repair.
Instead of just a spiral, we start with a spiral and then iteratively 
move each point towards the center of its largest adjacent Delaunay facet 
(the point of maximum distance to its neighbors). This directly targets 
the covering radius definition.
"""

import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

def _covering_radius(points):
    """
    Compute the covering radius in degrees.
    The covering radius is the maximum angular distance from any point on the sphere
    to its nearest configuration point. This is equivalent to the maximum circumradius
    of the facets of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        # Facets are indices of points forming the face
        facets = hull.simplices
        max_circum_r = 0.0
        
        # Precompute for speed? No, N is small.
        for i in range(len(facets)):
            p1, p2, p3 = points[facets[i][0]], points[facets[i][1]], points[facets[i][2]]
            
            # Calculate circumcenter of the spherical triangle
            # The circumcenter of a spherical triangle with vertices a, b, c
            # is proportional to (a x b) x (b x c) ... actually simpler:
            # It is the unit vector in the direction of the cross product of the normals?
            # No, the circumcenter is the point equidistant from a, b, c.
            # In 3D, the circumcenter of a triangle on the unit sphere is the unit vector
            # in the direction of the normal to the plane passing through a, b, c?
            # Wait, the plane passes through origin? No.
            # The circumcenter of a spherical triangle is the point on the sphere 
            # equidistant from the three vertices.
            # Let's compute it properly.
            # The center of the circle passing through a, b, c on the sphere.
            # The normal to the plane containing a,b,c is n = (b-a) x (c-a).
            # The circumcenter C satisfies (C-a).n = 0? No, C is on the plane? No.
            # C is the intersection of the sphere and the plane perpendicular to the 
            # chord midpoints?
            
            # Easier: The circumcenter of a spherical triangle is the unit vector 
            # in the direction of the cross product of the two edges? No.
            
            # Let's use the property that the circumcenter is proportional to 
            # (a x b) x (b x c) ... no.
            
            # Standard formula for circumcenter of spherical triangle:
            # If the triangle is not degenerate, the circumcenter is:
            # c = (a x b) x (b x c) ... no.
            
            # Let's solve for C such that |C-a| = |C-b| = |C-c| and |C|=1.
            # This is equivalent to C being in the plane defined by a,b,c? No.
            # The plane passing through a,b,c does not pass through the origin generally.
            
            # Actually, the circumcenter of a set of points on the sphere is the 
            # center of the smallest cap covering them.
            # For 3 points, it's the circumcenter of the spherical triangle.
            
            # Let's compute the normal to the plane defined by a, b, c.
            # n = cross(b-a, c-a)
            # The circumcenter C lies on the line passing through the origin? No.
            # The circumcenter C is the unit vector such that C is equidistant from a, b, c.
            # This means C is in the null space of the matrix formed by (a-b), (a-c) ?
            # No, C is the intersection of the sphere and the line perpendicular to the 
            # chord midpoints?
            
            # Let's use a robust numerical method for 3 points.
            # The circumcenter is the unit vector in the direction of:
            # (a * sin(2*angle_bc) + b * sin(2*angle_ca) + c * sin(2*angle_ab)) ?
            # No, that's for planar.
            
            # Let's just compute the circumcenter of the 3 points in 3D, 
            # then project to sphere.
            # The circumcenter of 3 points in 3D is the center of the circle passing through them.
            # This circle is NOT on the sphere generally, but the spherical circumcenter 
            # is the point on the sphere closest to the planar circumcenter?
            # No.
            
            # Let's use the cross product method which is standard for spherical geometry.
            # The circumcenter of a spherical triangle with vertices a, b, c is:
            # c = (a x b) x (b x c) ... no.
            
            # Let's derive it.
            # C is equidistant from a, b, c.
            # C . a = C . b = C . c = cos(theta)
            # So C is in the intersection of the planes:
            # (a-b).C = 0
            # (a-c).C = 0
            # And |C| = 1.
            
            # So C is parallel to (a-b) x (a-c).
            # Let's check.
            # n1 = a - b
            # n2 = a - c
            # C_dir = cross(n1, n2)
            # C = C_dir / |C_dir|
            
            n1 = p1 - p2
            n2 = p1 - p3
            c_dir = np.cross(n1, n2)
            
            if np.linalg.norm(c_dir) < 1e-10:
                # Degenerate, points are collinear or coincident
                # Fallback: use midpoint of the longest edge?
                # Or just skip?
                continue
                
            c = c_dir / np.linalg.norm(c_dir)
            
            # There are two possible circumcenters (c and -c).
            # We want the one inside the triangle.
            # Check if c is on the same side of the great circle arcs as the interior.
            # A simple check: c should have a positive dot product with the normal 
            # of the triangle face pointing outwards?
            # Actually, for a convex hull, the face normal points outwards.
            # The circumcenter should be "inside" the cap, so it should be 
            # closer to the face centroid than the antipode.
            
            # Let's check the dot product with the face normal.
            # Face normal is cross(p2-p1, p3-p1).
            f_normal = np.cross(p2 - p1, p3 - p1)
            f_normal = f_normal / np.linalg.norm(f_normal)
            
            # The circumcenter c should be in the direction of the face normal?
            # If the triangle is acute, the circumcenter is inside.
            # If obtuse, it's outside.
            # But for covering radius, we care about the empty cap.
            # The center of the empty cap is the circumcenter.
            # If the circumcenter is the antipode of the face normal, it's the "wrong" one.
            
            # Let's pick the one that is closer to the centroid of the face.
            centroid = (p1 + p2 + p3) / 3.0
            centroid = centroid / np.linalg.norm(centroid)
            
            if np.dot(c, centroid) < 0:
                c = -c
                
            # Now compute the angular distance from c to p1 (or p2, p3)
            # This is the circumradius.
            cos_theta = np.dot(c, p1)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            theta = np.arccos(cos_theta)
            
            if theta > max_circum_r:
                max_circum_r = theta
                
    except Exception:
        # If hull fails, fallback to brute force or return a large value
        # For now, return a large value to penalize bad configurations
        return np.pi
        
    return max_circum_r * 180.0 / np.pi

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Initial Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Optimization loop
    # We will iteratively move each point to the circumcenter of its worst facet.
    # This is a form of Lloyd's algorithm for spherical Voronoi tessellations.
    
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    for iter in range(1000):  # Max iterations
        if time.time() > deadline:
            break
            
        # Compute covering radius
        current_radius = _covering_radius(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
            
        # Find the point with the largest circumradius in its Delaunay facets
        # And move it to the circumcenter of that facet.
        
        try:
            hull = ConvexHull(points)
            facets = hull.simplices
        except Exception:
            # If hull fails, add some noise
            points += np.random.normal(0, 0.01, points.shape)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            continue
            
        # For each facet, compute circumcenter and circumradius
        # Find the facet with the largest circumradius
        max_rad = 0.0
        best_center = None
        best_facet_idx = -1
        
        for i in range(len(facets)):
            p1, p2, p3 = points[facets[i][0]], points[facets[i][1]], points[facets[i][2]]
            
            n1 = p1 - p2
            n2 = p1 - p3
            c_dir = np.cross(n1, n2)
            
            if np.linalg.norm(c_dir) < 1e-10:
                continue
                
            c = c_dir / np.linalg.norm(c_dir)
            
            centroid = (p1 + p2 + p3) / 3.0
            centroid = centroid / np.linalg.norm(centroid)
            
            if np.dot(c, centroid) < 0:
                c = -c
                
            cos_theta = np.dot(c, p1)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            theta = np.arccos(cos_theta)
            
            if theta > max_rad:
                max_rad = theta
                best_center = c
                best_facet_idx = i
                
        if best_center is not None:
            # Move the vertex of the worst facet that is "responsible" for the large radius?
            # Actually, moving any of the three vertices towards the center helps.
            # Let's move the one that is furthest from the center? Or just move all three slightly?
            # Standard Lloyd: move each site to the centroid of its cell.
            # Here, we are doing a "worst facet" move.
            
            # Let's move the vertex of the facet that is furthest from the circumcenter.
            # Or simply, move the entire configuration slightly?
            # No, let's move the specific vertex that is part of the worst facet.
            # Which one? Let's pick the one that is "most responsible".
            # Actually, a simpler heuristic: move the vertex that is furthest from the circumcenter.
            
            p1, p2, p3 = points[facets[best_facet_idx][0]], points[facets[best_facet_idx][1]], points[facets[best_facet_idx][2]]
            d1 = np.arccos(np.clip(np.dot(p1, best_center), -1.0, 1.0))
            d2 = np.arccos(np.clip(np.dot(p2, best_center), -1.0, 1.0))
            d3 = np.arccos(np.clip(np.dot(p3, best_center), -1.0, 1.0))
            
            max_d = max(d1, d2, d3)
            if max_d == d1:
                idx = facets[best_facet_idx][0]
            elif max_d == d2:
                idx = facets[best_facet_idx][1]
            else:
                idx = facets[best_facet_idx][2]
                
            # Move this point towards the circumcenter
            # Use a step size that is a fraction of the distance
            # To avoid oscillation, we can use a smaller step.
            # Let's move it completely to the center? That might cause overlap.
            # Let's move it 50% of the way.
            new_p = 0.5 * points[idx] + 0.5 * best_center
            new_p = new_p / np.linalg.norm(new_p)
            
            # Check if this move improves the local radius?
            # For now, just do it.
            points[idx] = new_p
            
    return best_points

# Ensure the function is named construct and returns the array
# The code above defines construct.

if __name__ == "__main__":
    # Simple test
    pts = construct()
    print(f"Covering radius: {_covering_radius(pts)} degrees")
    print(pts.shape)
