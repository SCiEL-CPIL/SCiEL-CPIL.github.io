<<<<<<< SEARCH
    def required_side_length(inner_centers, inner_angles, outer_center, outer_angle):
        """
        Compute the minimum side length S of outer hexagon (centered at outer_center, 
        rotated by outer_angle) that contains all inner hexagons.
        """
        # Outer hexagon has 6 sides. The support function of a regular hexagon with 
        # side length S centered at origin, flat-top, rotated by outer_angle:
        # The hexagon is the intersection of 6 half-planes.
        # For a regular hexagon with side S, the apothem (distance from center to side) is S * sqrt(3)/2.
        # The outward normals to the 6 sides are at angles outer_angle + k*60 for k=0..5.
        
        # For each of the 6 sides of the outer hexagon, we need:
        #   max over all inner hexagon vertices of (vertex - outer_center) @ normal_k <= S * sqrt(3)/2
        # for the side with normal normal_k pointing outward.
        
        # Actually, let me reconsider. A regular hexagon with side length S has:
        # - 6 sides, each at distance (apothem) = S * sqrt(3) / 2 from center
        # - The outward normals point at angles phi_k = outer_angle + k * 60 degrees
        
        # The constraint is that all inner hexagon vertices must lie inside the outer hexagon.
        # For side k with outward normal n_k, the constraint is:
        #   (v - c_outer) @ n_k <= apothem = S * sqrt(3) / 2
        # for all vertices v of all inner hexagons.
        
        # So S >= max over all k and all v of: (v - c_outer) @ n_k / (sqrt(3)/2)
        #    = max over all k and all v of: 2/sqrt(3) * (v - c_outer) @ n_k
        
        # Also need S > 0 and the inner hexagons must be disjoint from each other.
        
        oc = np.array(outer_center)
        oa = outer_angle * np.pi / 180.0
        
        # Normals of outer hexagon sides
        normal_angles = oa + np.arange(6) * np.pi / 3
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        apothem_factor = np.sqrt(3.0) / 2.0
        
        max_val = 0.0
        for idx in range(N):
            center = np.array(inner_centers[idx])
            angle = inner_angles[idx] * np.pi / 180.0
            verts = rotate_vertices(angle) + center
            # For each normal, compute max dot product
            dots = verts @ normals.T  # shape (6, 6)
            for k in range(6):
                val = np.max(dots[:, k])
                if val > max_val:
                    max_val = val
        
        S = max_val / apothem_factor
        if S < 0:
            S = 0.0
        return S
=======
    def required_side_length(inner_centers, inner_angles, outer_center, outer_angle):
        """
        Compute the minimum side length S of outer hexagon (centered at outer_center, 
        rotated by outer_angle) that contains all inner hexagons.
        """
        oc = np.array(outer_center)
        oa = outer_angle * np.pi / 180.0
        
        # Normals of outer hexagon sides
        normal_angles = oa + np.arange(6) * np.pi / 3
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        apothem_factor = np.sqrt(3.0) / 2.0
        
        max_val = 0.0
        for idx in range(N):
            center = np.array(inner_centers[idx])
            angle = inner_angles[idx] * np.pi / 180.0
            verts = rotate_vertices(angle) + center
            # For each normal, compute max dot product of (vertex - outer_center) with normal
            relative_verts = verts - oc
            # dots shape: (6 vertices, 6 normals)
            dots = relative_verts @ normals.T
            for k in range(6):
                val = np.max(dots[:, k])
                if val > max_val:
                    max_val = val
        
        S = max_val / apothem_factor
        if S < 0:
            S = 0.0
        return S
>>>>>>> REPLACE

