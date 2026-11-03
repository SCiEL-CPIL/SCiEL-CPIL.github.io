    def objective(x):
        pts = x.reshape(N, 3)
        # Normalize
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        # If any norm is 0, return inf
        if np.any(norms < 1e-8): return 1e10
        pts = pts / norms
        
        try:
            hull = ConvexHull(pts)
        except Exception:
            return 1e10
            
        # Check if origin is inside
        # The convex hull of points on the sphere contains the origin iff the points are not in any open hemisphere.
        # We can check this by verifying that the normal of each facet points away from the origin?
        # Or simpler: if the hull is valid, check the volume? No, it's a polyhedron.
        # A robust check: The origin is inside if the signed volume of the tetrahedron formed by origin and each facet is positive (or all same sign).
        # Actually, scipy's ConvexHull provides `simplices`.
        # For each simplex (triangle), compute the normal. If the dot product of the normal and the midpoint is positive, it points outward.
        # If the origin is inside, the origin should be on the "inner" side of all facets.
        # Let's just rely on the fact that if they are in a hemisphere, the hull will be a "cap" and the circumradius calculation might be weird or the hull might not enclose the origin.
        # Actually, if points are in a hemisphere, the convex hull is a subset of the sphere surface, not a polyhedron enclosing the origin.
        # The "covering radius" definition implies we want the points to cover the sphere, so they must not be in a hemisphere.
        # If they are in a hemisphere, the covering radius is > 90 deg.
        # We can detect this: if the origin is not in the convex hull, return a large penalty.
        
        # Check if origin is in hull:
        # For each facet, the plane equation is n . x = d.
        # If n . 0 < d for all facets (assuming n points outward), then origin is outside?
        # Let's use a simpler check: The sum of areas of the projection of the hull onto a hemisphere?
        # Simpler: Just compute the circumradii. If the points are in a hemisphere, the "hull" is not closed around the origin.
        # However, ConvexHull will still compute the convex hull of the points.
        # The circumradii of the facets of this hull will be small (since the points are clustered).
        # BUT the covering radius of the sphere is large.
        # So minimizing the sum of circumradii of the hull facets will drive points to cluster, which is BAD for covering.
        # We MUST ensure the points are well-distributed.
        # Penalty: If the origin is not inside the convex hull, add a large penalty.
        
        # Check origin containment:
        # The origin is inside the convex hull if and only if there is no plane separating the origin from all points.
        # Equivalent: The maximum angle between any point and the center of the hull?
        # Let's use the property: The origin is inside the convex hull of $P$ iff there exist $\lambda_i \ge 0, \sum \lambda_i = 1$ such that $\sum \lambda_i p_i = 0$.
        # This is a linear programming problem. Too slow for every objective call.
        
        # Heuristic: If the points are in a hemisphere, the "circumradii" of the hull facets will be small, but the *covering* is bad.
        # Wait, if points are in a hemisphere, the convex hull is a "dome". The facets are on the "top".
        # The circumradius of these facets is small.
        # So the objective $\sum R^2$ will be small.
        # This means the optimizer will try to cluster points!
        # We need to prevent this.
        # We can add a term that penalizes the "solid angle" covered by the hull?
        # Or simply: If the points are in a hemisphere, the covering radius is at least 90 deg.
        # We can check if the points span the sphere.
        # A simple check: Compute the convex hull. If the origin is not inside, return a penalty based on the distance of the origin to the hull.
        
        # Distance from origin to hull:
        # For each facet, compute the distance from origin to the plane.
        # If the origin is outside, the minimum distance to the hull is the min of these distances (for facets facing the origin).
        # Let's implement a quick check:
        # For each simplex, get the 3 vertices.
        # Compute the normal vector.
        # Compute the distance from origin to the plane.
        # If the origin is on the "wrong" side of any facet, it's outside.
        
        # Actually, there is a simpler metric: The covering radius is bounded by the circumradius of the *largest* empty circle.
        # If the points are in a hemisphere, the largest empty circle is the rest of the sphere, radius 90 deg.
        # So we should return 90 deg (or pi/2) as the score if the origin is not inside the hull.
        
        # How to check if origin is inside?
        # The convex hull contains the origin if the origin is on the same side of all facet planes as the interior.
        # For a facet with vertices $v_1, v_2, v_3$, the normal $n = (v_2-v_1) \times (v_3-v_1)$.
        # The plane is $n \cdot x = n \cdot v_1$.
        # The origin $0$ satisfies $n \cdot 0 = 0$.
        # If $n \cdot v_1 > 0$, the origin is on the negative side.
        # If all facets have $n \cdot v_i > 0$ (assuming outward normals), then the origin is inside.
        # If any facet has $n \cdot v_i < 0$, the origin is outside.
        # Note: The orientation of the normal from `ConvexHull` is outward.
        # So if $n \cdot v_1 < 0$ for any facet, the origin is outside.
        
        # Let's implement this check.
        
        radii = []
        origin_inside = True
        for simplex in hull.simplices:
            v1, v2, v3 = pts[simplex]
            # Side lengths
            a = np.linalg.norm(v2 - v1)
            b = np.linalg.norm(v3 - v2)
            c = np.linalg.norm(v1 - v3)
            
            # Area
            cross = np.cross(v2 - v1, v3 - v1)
            area = 0.5 * np.linalg.norm(cross)
            
            if area < 1e-8:
                continue
                
            R = (a * b * c) / (4 * area)
            radii.append(R)
            
            # Check origin
            # Normal is cross product.
            # Is it outward?
            # The centroid of the facet is (v1+v2+v3)/3.
            # Vector from origin to centroid is c.
            # If dot(n, c) > 0, normal points outward.
            # If dot(n, c) < 0, normal points inward.
            # ConvexHull normals are outward.
            # So dot(n, c) should be > 0.
            # The origin is inside if dot(n, 0) < dot(n, c) ?
            # Plane eq: n . x = n . c.
            # Origin: n . 0 = 0.
            # If 0 < n . c, origin is on the "inside" (assuming n points away from inside).
            # Wait, if n points outward, the interior is n . x < n . c.
            # So if 0 < n . c, origin is inside.
            # If 0 > n . c, origin is outside.
            
            dot_c = np.dot(cross, v1) # n . v1
            if dot_c < 0:
                origin_inside = False
                
        if not origin_inside:
            # Penalty: The covering radius is at least the angular distance to the closest point to the "gap".
            # Just return a large value, e.g., pi/2 (90 deg)
            return np.pi/2
            
        if not radii:
            return 1e10
            
        return np.sum(np.array(radii)**2)
    