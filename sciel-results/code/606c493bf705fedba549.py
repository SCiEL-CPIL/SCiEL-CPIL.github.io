"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_pts = pts.copy()
    best_radius = np.inf
    
    # Helper to compute covering radius using convex hull
    def get_covering_radius(points):
        try:
            hull = ConvexHull(points)
            # Facets are in hull.simplices
            # We need the circumradius of each facet (triangle)
            # Since points are on unit sphere, the circumradius R of a triangle with side lengths a,b,c
            # and area A is R = abc / (4A).
            # Alternatively, the distance from origin to the plane of the facet is 1/R if origin is inside?
            # No, for a unit sphere, the angular radius theta is given by cos(theta) = h, where h is distance from origin to plane.
            # Actually, the cap radius is the angle subtended by the facet.
            # For a triangle with vertices v1, v2, v3 on unit sphere, the circumcenter of the triangle (on the sphere) is the point furthest from the origin in the plane? No.
            # The point on the sphere furthest from the triangle vertices is the "pole" of the facet normal.
            # The angular distance from the pole to any vertex is the circumradius.
            # If the facet normal is n, the pole is n (if origin is inside).
            # The distance from n to a vertex v is arccos(n . v).
            # So we just need to compute the normal of each facet and find the max angle to its vertices.
            
            max_ang = 0.0
            for simplex in hull.simplices:
                v1 = points[simplex[0]]
                v2 = points[simplex[1]]
                v3 = points[simplex[2]]
                
                # Compute normal
                edge1 = v2 - v1
                edge2 = v3 - v1
                normal = np.cross(edge1, edge2)
                norm = np.linalg.norm(normal)
                if norm < 1e-10:
                    continue
                normal = normal / norm
                
                # Ensure normal points outward (away from origin)
                if np.dot(normal, v1) < 0:
                    normal = -normal
                    
                # The angular radius is the max angle between normal and vertices
                # Actually, if the origin is inside, the "cap" center is the normal.
                # The distance to vertices is arccos(dot(normal, vi)).
                # Since it's a triangle on the sphere, the circumcenter is the normal direction?
                # Yes, for a spherical triangle, the circumcenter is the intersection of perpendicular bisectors, which lies on the normal to the plane?
                # Wait, the circumcenter of a spherical triangle is the point equidistant from the 3 vertices.
                # For a triangle on the unit sphere, the circumcenter is indeed the normalized cross product of the edges? No.
                # The circumcenter $c$ satisfies $c \cdot v_1 = c \cdot v_2 = c \cdot v_3$.
                # This implies $c$ is parallel to the normal of the plane containing $v_1, v_2, v_3$.
                # So yes, $c = \pm normal$.
                
                d1 = np.dot(normal, v1)
                d2 = np.dot(normal, v2)
                d3 = np.dot(normal, v3)
                
                # Clamp for numerical stability
                d1 = np.clip(d1, -1.0, 1.0)
                d2 = np.clip(d2, -1.0, 1.0)
                d3 = np.clip(d3, -1.0, 1.0)
                
                ang1 = np.arccos(d1)
                ang2 = np.arccos(d2)
                ang3 = np.arccos(d3)
                
                # The covering radius for this facet is the max of these angles?
                # No, the circumradius is the distance from the circumcenter to the vertices.
                # Since the circumcenter is equidistant, ang1 should equal ang2 and ang3.
                # Due to floating point errors, they might differ slightly.
                # The "cap" defined by this facet covers the region around the normal.
                # The radius of the cap is the angular distance to the vertices.
                
                ang = max(ang1, ang2, ang3)
                if ang > max_ang:
                    max_ang = ang
                    
            return max_ang
        except Exception:
            return np.pi # Worst case
            
    current_radius = get_covering_radius(pts)
    if current_radius < best_radius:
        best_radius = current_radius
        best_pts = pts.copy()
        
    step_size = 0.05
    learning_rate = 1.0
    
    while time.time() < deadline:
        # Find the worst facet
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
            
        worst_facet_idx = -1
        max_ang = -1.0
        worst_normal = None
        
        for idx, simplex in enumerate(hull.simplices):
            v1 = pts[simplex[0]]
            v2 = pts[simplex[1]]
            v3 = pts[simplex[2]]
            
            edge1 = v2 - v1
            edge2 = v3 - v1
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm < 1e-10:
                continue
            normal = normal / norm
            if np.dot(normal, v1) < 0:
                normal = -normal
            
            d1 = np.clip(np.dot(normal, v1), -1.0, 1.0)
            d2 = np.clip(np.dot(normal, v2), -1.0, 1.0)
            d3 = np.clip(np.dot(normal, v3), -1.0, 1.0)
            
            ang = max(np.arccos(d1), np.arccos(d2), np.arccos(d3))
            
            if ang > max_ang:
                max_ang = ang
                worst_facet_idx = idx
                worst_normal = normal
        
        if worst_facet_idx == -1:
            break
            
        # Get the vertices of the worst facet
        simplex = hull.simplices[worst_facet_idx]
        v1 = pts[simplex[0]]
        v2 = pts[simplex[1]]
        v3 = pts[simplex[2]]
        
        # Move these points towards the worst_normal (the center of the hole)
        # This reduces the cap radius.
        # We should also consider moving them away from each other? No, towards the center is the direct fix.
        
        alpha = step_size * learning_rate
        
        # Update points
        pts[simplex[0]] = v1 + alpha * worst_normal
        pts[simplex[1]] = v2 + alpha * worst_normal
        pts[simplex[2]] = v3 + alpha * worst_normal
        
        # Normalize
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        
        # Check new radius
        new_radius = get_covering_radius(pts)
        
        if new_radius < current_radius:
            current_radius = new_radius
            if current_radius < best_radius:
                best_radius = current_radius
                best_pts = pts.copy()
            # Decay step size slowly
            step_size *= 0.99
        else:
            # Revert
            pts[simplex[0]] = v1
            pts[simplex[1]] = v2
            pts[simplex[2]] = v3
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / norms
            
            # Reduce step size
            step_size *= 0.9
            if step_size < 1e-5:
                break
                
    return best_pts
# EVOLVE-BLOCK-END
