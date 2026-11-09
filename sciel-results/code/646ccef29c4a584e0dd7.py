import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # 1. Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_pts = pts.copy()
    best_radius = np.inf
    
    # Helper to compute covering radius from hull
    def get_covering_radius(points):
        try:
            hull = ConvexHull(points)
            max_r = 0.0
            for simplex in hull.simplices:
                v1 = points[simplex[0]]
                v2 = points[simplex[1]]
                v3 = points[simplex[2]]
                
                # Circumcenter of spherical triangle
                # Cross product gives normal
                n = np.cross(v1, v2)
                n = n / np.linalg.norm(n)
                
                # Circumcenter is in direction n (or -n)
                # Check which side is the "outside"
                # The centroid of the triangle should be on the same side as the circumcenter?
                # Actually, for a convex hull facet, the normal points outward.
                # The circumcenter of the spherical triangle is the point on the sphere
                # equidistant from v1, v2, v3.
                # It lies along the normal to the plane of the triangle.
                
                c = n
                # Ensure c is "outside" the triangle in terms of dot products?
                # The distance from c to v1 is arccos(dot(c, v1)).
                # We want the c such that the triangle is "contained" in the cap.
                # If dot(c, v1) > 0, the angle is < 90.
                
                d1 = np.arccos(np.clip(np.dot(c, v1), -1, 1))
                d2 = np.arccos(np.clip(np.dot(c, v2), -1, 1))
                d3 = np.arccos(np.clip(np.dot(c, v3), -1, 1))
                
                # If the triangle is obtuse, the circumcenter might be "far" or the max distance is to a vertex?
                # For covering radius, we look for the point in the Voronoi cell furthest from the vertices.
                # If the spherical triangle is acute, the circumcenter is inside, and it is the furthest point.
                # If it is obtuse, the circumcenter is outside the triangle (in the sense that the great circle arc passes through it? No).
                # Actually, the Voronoi cell is the set of points closer to v1 than others.
                # The vertices of the Voronoi cell are the circumcenters of the facets.
                # The covering radius is the max distance from any point in the cell to the nearest vertex.
                # This is equivalent to the max circumradius of the facets, *provided* the circumcenter lies within the facet.
                # If the circumcenter lies outside the facet, the max distance is to the midpoint of the longest edge.
                
                # Let's check if c is inside the spherical triangle.
                # A point c is inside if it is on the same side of each edge as the opposite vertex.
                # Edge v1-v2: normal is cross(v1, v2). Check dot(cross(v1,v2), v3) and dot(cross(v1,v2), c).
                # If signs differ, c is outside.
                
                n12 = np.cross(v1, v2)
                n23 = np.cross(v2, v3)
                n31 = np.cross(v3, v1)
                
                inside = True
                if np.dot(n12, v3) * np.dot(n12, c) < 0: inside = False
                if np.dot(n23, v1) * np.dot(n23, c) < 0: inside = False
                if np.dot(n31, v2) * np.dot(n31, c) < 0: inside = False
                
                if inside:
                    r_circ = d1 # d1==d2==d3
                else:
                    # Max distance is to the midpoint of the longest edge
                    d12 = np.arccos(np.clip(np.dot(v1, v2), -1, 1))
                    d23 = np.arccos(np.clip(np.dot(v2, v3), -1, 1))
                    d31 = np.arccos(np.clip(np.dot(v3, v1), -1, 1))
                    max_edge = max(d12, d23, d31)
                    r_circ = max_edge / 2.0
                
                if r_circ > max_r:
                    max_r = r_circ
            return max_r
        except Exception:
            return np.pi # Fallback if hull fails

    # Initial evaluation
    current_radius = get_covering_radius(pts)
    if current_radius < best_radius:
        best_radius = current_radius
        best_pts = pts.copy()

    # Iterative improvement
    step_size = 0.05
    no_improve_count = 0
    
    while time.time() < deadline:
        # Find worst facet
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
            
        worst_idx = -1
        worst_r = 0
        worst_c = None
        worst_v = None
        
        for i_s, simplex in enumerate(hull.simplices):
            v1 = pts[simplex[0]]
            v2 = pts[simplex[1]]
            v3 = pts[simplex[2]]
            
            n = np.cross(v1, v2)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-10: continue
            n = n / norm_n
            
            # Determine correct orientation
            # The normal should point away from the origin? No, convex hull normals point outward.
            # But we need the circumcenter.
            # Let's just use n and -n and pick the one that gives smaller distances?
            # Actually, for a convex hull of points on the sphere, the facet normal points outward.
            # The circumcenter of the spherical triangle is in the direction of the normal if the triangle is "acute" relative to the center.
            
            c = n
            # Check if c is "close" to the vertices
            d1 = np.dot(c, v1)
            d2 = np.dot(c, v2)
            d3 = np.dot(c, v3)
            
            # If any dot product is negative, the angle is > 90, so c might be on the "other side"
            # In that case, the circumcenter of the *cell* might be different or the max distance is determined by edges.
            
            # Let's compute the circumradius properly
            # Distance from c to v1
            r1 = np.arccos(np.clip(d1, -1, 1))
            
            # Check if c is inside the triangle
            n12 = np.cross(v1, v2)
            n23 = np.cross(v2, v3)
            n31 = np.cross(v3, v1)
            
            inside = True
            if np.dot(n12, v3) * np.dot(n12, c) < 0: inside = False
            if np.dot(n23, v1) * np.dot(n23, c) < 0: inside = False
            if np.dot(n31, v2) * np.dot(n31, c) < 0: inside = False
            
            if inside:
                r_facet = r1
            else:
                d12 = np.arccos(np.clip(np.dot(v1, v2), -1, 1))
                d23 = np.arccos(np.clip(np.dot(v2, v3), -1, 1))
                d31 = np.arccos(np.clip(np.dot(v3, v1), -1, 1))
                max_edge = max(d12, d23, d31)
                r_facet = max_edge / 2.0
            
            if r_facet > worst_r:
                worst_r = r_facet
                worst_idx = i_s
                worst_c = c
                worst_v = simplex

        if worst_idx == -1:
            break
            
        # Move the vertex with the largest angle to the circumcenter?
        # Or just move one vertex towards the center.
        # Let's move the vertex that is "most central" or simply rotate the whole triangle?
        # Moving one vertex is simpler. Let's move the first one.
        # Better: Move all three towards the center?
        
        # Update pts
        for idx in worst_v:
            old_pt = pts[idx]
            # Move towards worst_c
            new_pt = old_pt + step_size * (worst_c - old_pt)
            pts[idx] = new_pt / np.linalg.norm(new_pt)
            
        # Evaluate
        current_radius = get_covering_radius(pts)
        if current_radius < best_radius:
            best_radius = current_radius
            best_pts = pts.copy()
            no_improve_count = 0
            # Reduce step size slightly
            step_size *= 0.95
        else:
            no_improve_count += 1
            # Revert if no improvement for a while? Or just keep going?
            # If we revert, we might oscillate.
            # Let's just keep the current pts and try a different step or vertex.
            # For simplicity, we'll just continue.
            
        # If no improvement for many steps, reduce step size
        if no_improve_count > 100:
            step_size *= 0.5
            no_improve_count = 0
            
    return best_pts
