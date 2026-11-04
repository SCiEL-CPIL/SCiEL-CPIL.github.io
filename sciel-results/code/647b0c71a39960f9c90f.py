import numpy as np
import time
from scipy.spatial import ConvexHull
import sys

N = 130

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Leave 10s buffer for safety
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Helper to normalize
    def normalize(P):
        norms = np.linalg.norm(P, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1.0
        return P / norms

    # Helper to compute covering radius and worst facet
    def get_covering_info(points):
        try:
            hull = ConvexHull(points)
        except Exception:
            # If hull fails, return large radius
            return np.inf, None, None
            
        max_r = 0.0
        worst_face = None
        worst_circumcenter = None
        
        # Iterate over faces
        for face in hull.simplices:
            v1 = points[face[0]]
            v2 = points[face[1]]
            v3 = points[face[2]]
            
            # Compute spherical circumcenter
            # The circumcenter is the intersection of the perpendicular bisectors.
            # Alternatively, solve for a point c such that |c-v1|^2 = |c-v2|^2 = |c-v3|^2 and |c|=1.
            # This implies c is in the plane defined by v1, v2, v3? No, it's on the sphere.
            # The circumcenter of a spherical triangle is the normalized vector:
            # c = (v1 * sin(A) + v2 * sin(B) + v3 * sin(C)) / norm(...) ?
            # Or simpler:
            # The circumcenter is the unit vector in the direction of:
            # v1 * (v2 x v3) + v2 * (v3 x v1) + v3 * (v1 x v2)
            # Wait, that's the centroid of the triangle on the sphere? No.
            
            # Let's use the property that the circumcenter c satisfies:
            # c . v1 = c . v2 = c . v3 = cos(r)
            # So c is the solution to:
            # M c = d, where M is 3x3 matrix of v1, v2, v3 as rows, d = [cos(r), cos(r), cos(r)]
            # But we don't know r.
            # However, c is proportional to the vector:
            # c_unnorm = v1 * (v2 x v3) + v2 * (v3 x v1) + v3 * (v1 x v2)
            # Let's verify this.
            # If c = v1 x v2 x v3 ... no.
            
            # Standard formula for circumcenter of 3 points on sphere:
            # c = (v1 * sin(2*A) + v2 * sin(2*B) + v3 * sin(2*C)) ?
            
            # Easier numerical approach:
            # The circumcenter is the point on the sphere equidistant to v1, v2, v3.
            # It lies in the plane perpendicular to the normal of the triangle? No.
            # It lies in the plane defined by the origin and the two bisectors.
            
            # Let's solve:
            # c = a*v1 + b*v2 + c*v3
            # This is not necessarily true.
            
            # Correct method:
            # The circumcenter is the normalized cross product of the two bisector vectors?
            # Bisector of v1, v2 is in direction (v1+v2) if they are close? No.
            
            # Let's use the formula:
            # c = (v1 * (v2 x v3) + v2 * (v3 x v1) + v3 * (v1 x v2))
            # This vector is perpendicular to the plane of the triangle? No.
            
            # Let's use `scipy.spatial` or linear algebra.
            # The circumcenter c satisfies:
            # c^T v1 = c^T v2 = c^T v3
            # Let c = x * v1 + y * v2 + z * v3 ? No, c is not necessarily in the span of v1,v2,v3 in R3?
            # Actually, the circumcenter of a triangle on a sphere IS in the span of the vertices in R3.
            # Why? The set of points equidistant to v1 and v2 is a plane through the origin.
            # The set of points equidistant to v2 and v3 is a plane through the origin.
            # Their intersection is a line through the origin.
            # So c is a scalar multiple of the cross product of the normals of these two planes.
            
            # Plane 1: points equidistant to v1, v2.
            # x^T v1 = x^T v2 => x^T (v1 - v2) = 0.
            # Normal n1 = v1 - v2.
            # Plane 2: points equidistant to v2, v3.
            # x^T (v2 - v3) = 0.
            # Normal n2 = v2 - v3.
            
            # c is parallel to n1 x n2.
            n1 = v1 - v2
            n2 = v2 - v3
            c_vec = np.cross(n1, n2)
            
            if np.linalg.norm(c_vec) < 1e-10:
                # Degenerate, skip or handle
                continue
                
            c = c_vec / np.linalg.norm(c_vec)
            
            # Check if c is the "internal" circumcenter.
            # The distance r is arccos(c . v1).
            dot = np.clip(np.dot(c, v1), -1.0, 1.0)
            r = np.arccos(dot)
            
            # We must ensure c is the center of the cap that covers the triangle.
            # If the triangle is "large" (contains the antipode of c?), the circumcenter might be the antipode.
            # The covering radius for the facet is the minimum of r and pi - r?
            # No, the facet is a small cap. The vertices are on the sphere.
            # The circumcenter c is the center of the circle passing through v1, v2, v3.
            # There are two such centers (c and -c).
            # We want the one that is "close" to the triangle.
            # The triangle is the convex hull facet, so it is "small" (less than a hemisphere).
            # So c should be such that the angle between c and the vertices is < pi/2?
            # Actually, if the triangle is obtuse, the circumcenter might be outside the triangle?
            # On a sphere, the "circumcenter" of the Voronoi cell is the point in the cell furthest from the vertices.
            # For a spherical triangle, this is the circumcenter if it lies inside the triangle.
            # If the circumcenter lies outside, the furthest point is a vertex.
            
            # How to check if c is inside the spherical triangle?
            # Check if the angles between c and the vertices are consistent.
            # Or simpler: The circumradius of the Voronoi cell is the distance to the nearest vertex.
            # If c is the "wrong" pole (antipode), the distance is large.
            # We want the minimum covering radius, so we should pick the c that minimizes the max distance to vertices?
            # Actually, the Voronoi cell is the set of points closer to v1 than to others.
            # The circumcenter of the facet is the point equidistant to v1, v2, v3.
            # If this point is inside the cell, it is the worst point.
            # If it is outside, the worst point is a vertex.
            
            # Let's compute the distance from c to v1.
            # If c is the "internal" center, the distance is r.
            # If c is the "external" center, the distance is pi - r.
            # We want the center that is inside the triangle.
            # A simple check: The dot product of c with the normal of the triangle (v1 x v2) should have the same sign as the dot product of the centroid (v1+v2+v3)/3 with the normal?
            # Or just: The circumcenter c should be such that the angles from c to v1, v2, v3 are all < pi/2?
            # If any angle > pi/2, the triangle is "obtuse" and the circumcenter is outside?
            
            # Let's just use the distance r.
            # If r > pi/2, we should probably use the antipode -c, which has distance pi - r.
            # But wait, the Voronoi cell is defined by the nearest point.
            # If the triangle is large, the "cap" might wrap around.
            # For N=130, the facets are small. r should be small (~11 degrees).
            # So r < pi/2 is guaranteed.
            
            if r > max_r:
                max_r = r
                worst_face = face
                worst_circumcenter = c
                
        return max_r, worst_face, worst_circumcenter

    # Main optimization loop
    current_points = points
    best_points = points.copy()
    best_radius, _, _ = get_covering_info(best_points)
    
    step_size = 0.05
    
    while time.time() < deadline:
        current_radius, face, c_center = get_covering_info(current_points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = current_points.copy()
            
        if face is None or c_center is None:
            break
            
        # Move the vertices of the worst face towards the circumcenter
        v1 = current_points[face[0]]
        v2 = current_points[face[1]]
        v3 = current_points[face[2]]
        
        # Update
        # We want to reduce the distance from c_center to the vertices.
        # Moving vertices towards c_center reduces the distance.
        # But we must keep them on the sphere.
        
        new_v1 = v1 + step_size * (c_center - v1)
        new_v2 = v2 + step_size * (c_center - v2)
        new_v3 = v3 + step_size * (c_center - v3)
        
        # Normalize
        new_v1 = new_v1 / np.linalg.norm(new_v1)
        new_v2 = new_v2 / np.linalg.norm(new_v2)
        new_v3 = new_v3 / np.linalg.norm(new_v3)
        
        current_points[face[0]] = new_v1
        current_points[face[1]] = new_v2
        current_points[face[2]] = new_v3
        
        # Optional: Add small random perturbation to escape local minima
        # But let's stick to deterministic first.
        
        # Check time
        if time.time() > deadline - 5.0:
            break
            
    # Final check
    final_radius, _, _ = get_covering_info(best_points)
    if final_radius < best_radius:
        best_radius = final_radius
        
    return best_points

# EVOLVE-BLOCK-END
