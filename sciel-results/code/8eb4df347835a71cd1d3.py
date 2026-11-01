"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import math


def _covering_radius(points):
    """Compute the covering radius (in radians) as the max circumradius of convex hull facets."""
    try:
        hull = ConvexHull(points)
        max_r = 0.0
        for simplex in hull.simplices:
            # Get the 3 points of the facet (triangle)
            p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            # Circumradius of a triangle on the unit sphere
            # Using the formula: circumradius R = (a*b*c) / (4*Area) where a,b,c are side lengths
            # But for spherical triangles, we need the spherical circumradius
            # Alternatively, find the circumcenter of the triangle on the sphere
            # The circumcenter is the point on the sphere equidistant from p1, p2, p3
            # For a triangle on the unit sphere, the circumradius (angular) can be found by:
            # R = arccos( (1 - (a^2+b^2+c^2)/2) / sqrt(1 - a^2) ... ) -- this is complex
            # 
            # Better approach: The circumcenter C of triangle p1,p2,p3 on the unit sphere
            # satisfies C·p1 = C·p2 = C·p3 = cos(R)
            # So C is in the span of p1,p2,p3 and |C|=1
            # C = alpha*p1 + beta*p2 + gamma*p3
            # C·p1 = C·p2 = C·p3
            # 
            # Actually, simpler: the circumcenter of a spherical triangle is the intersection
            # of the sphere with the line through the origin perpendicular to the plane of the triangle?
            # No, that's only for right triangles.
            #
            # Let's use the formula for the circumradius of a spherical triangle.
            # For a triangle with side lengths a, b, c (angular distances),
            # the circumradius R satisfies:
            # sin(R) = (sin(a) * sin(b) * sin(c)) / (2 * sin(s-a) * sin(s-b) * sin(s-c))
            # where s = (a+b+c)/2 ... actually this is for the area.
            #
            # Let me use a direct geometric approach:
            # The circumcenter C of triangle p1,p2,p3 on the unit sphere is the unit vector
            # in the direction of the cross product of (p2-p1) and (p3-p1) ... no.
            #
            # Actually, the circumcenter is the point C on the unit sphere such that
            # |C - p1| = |C - p2| = |C - p3| (Euclidean distance).
            # This means C is equidistant from p1, p2, p3 in R^3.
            # So C lies on the line perpendicular to the plane containing p1,p2,p3 through their circumcenter in that plane.
            # 
            # Let me compute:
            # Let v = p2 - p1, w = p3 - p1
            # The normal to the plane is n = cross(v, w)
            # The circumcenter in the plane of p1,p2,p3 is:
            # C_plane = p1 + t1*v + t2*w
            # where t1, t2 solve:
            # (C_plane - p1)·v = |v|^2/2  =>  t1|v|^2 + t2(v·w) = |v|^2/2
            # (C_plane - p1)·w = |w|^2/2  =>  t1(v·w) + t2|w|^2 = |w|^2/2
            #
            # Then the circumcenter on the sphere is in the direction of C_plane (from origin)
            # because the set of points equidistant from p1,p2,p3 is a line perpendicular to the plane.
            # The point on the unit sphere on this line is C = C_plane / |C_plane|
            # The angular distance from C to p1 is arccos(C·p1)
            
            v = p2 - p1
            w = p3 - p1
            vv = np.dot(v, v)
            ww = np.dot(w, w)
            vw = np.dot(v, w)
            det = vv * ww - vw * vw
            if abs(det) < 1e-12:
                # Degenerate triangle, skip
                continue
            t1 = (0.5 * vv * ww - 0.5 * ww * vw) / det
            t2 = (0.5 * ww * vv - 0.5 * vv * vw) / det
            C_plane = p1 + t1 * v + t2 * w
            norm_C = np.linalg.norm(C_plane)
            if norm_C < 1e-12:
                continue
            C = C_plane / norm_C
            # Ensure C is on the correct side (same hemisphere as the triangle center)
            # The angular distance
            cos_R = np.clip(np.dot(C, p1), -1.0, 1.0)
            R = np.arccos(cos_R)
            max_r = max(max_r, R)
        return max_r
    except Exception:
        return np.pi  # Return large value on error


def _optimize_covering(points):
    """Optimize the covering radius by moving points to reduce the max facet circumradius."""
    points = points.copy()
    
    # Identify the facets with the largest circumradius and move points to reduce them
    for iteration in range(200):
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        # Compute circumradius for each facet
        facet_radii = []
        for simplex in hull.simplices:
            p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            v = p2 - p1
            w = p3 - p1
            vv = np.dot(v, v)
            ww = np.dot(w, w)
            vw = np.dot(v, w)
            det = vv * ww - vw * vw
            if abs(det) < 1e-12:
                facet_radii.append(0.0)
                continue
            t1 = (0.5 * vv * ww - 0.5 * ww * vw) / det
            t2 = (0.5 * ww * vv - 0.5 * vv * vw) / det
            C_plane = p1 + t1 * v + t2 * w
            norm_C = np.linalg.norm(C_plane)
            if norm_C < 1e-12:
                facet_radii.append(0.0)
                continue
            C = C_plane / norm_C
            cos_R = np.clip(np.dot(C, p1), -1.0, 1.0)
            R = np.arccos(cos_R)
            facet_radii.append(R)
        
        facet_radii = np.array(facet_radii)
        max_idx = np.argmax(facet_radii)
        max_r = facet_radii[max_idx]
        
        if max_r < 0.197:  # ~11.3 degrees in radians
            break
        
        # Get the points of the worst facet
        simplex = hull.simplices[max_idx]
        idx1, idx2, idx3 = simplex
        
        # Move the points of this facet slightly outward (toward the circumcenter)
        # to reduce the circumradius
        p1, p2, p3 = points[idx1], points[idx2], points[idx3]
        v = p2 - p1
        w = p3 - p1
        vv = np.dot(v, v)
        ww = np.dot(w, w)
        vw = np.dot(v, w)
        det = vv * ww - vw * vw
        if abs(det) < 1e-12:
            break
        t1 = (0.5 * vv * ww - 0.5 * ww * vw) / det
        t2 = (0.5 * ww * vv - 0.5 * vv * vw) / det
        C_plane = p1 + t1 * v + t2 * w
        norm_C = np.linalg.norm(C_plane)
        if norm_C < 1e-12:
            break
        C = C_plane / norm_C
        
        # Move each point slightly toward the circumcenter
        step = 0.001
        for idx in [idx1, idx2, idx3]:
            p = points[idx]
            new_p = p + step * (C - p)
            new_p = new_p / np.linalg.norm(new_p)
            points[idx] = new_p
    
    return points


def construct():
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Optimize to reduce covering radius
    points = _optimize_covering(points)
    
    # Final normalization
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    return points
# EVOLVE-BLOCK-END
