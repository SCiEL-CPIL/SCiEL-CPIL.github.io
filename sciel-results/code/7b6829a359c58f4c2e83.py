<<<<<<< SEARCH
    def required_side_length(inner_positions, outer_center=(0.0, 0.0), outer_angle_deg=0.0):
        """Compute minimum side length of outer hexagon to contain all inner hexagons."""
        ocx, ocy = outer_center
        oa = np.radians(outer_angle_deg)
        
        max_val = 0.0
        for (cx, cy, angle_deg) in inner_positions:
            verts = hex_vertices(cx, cy, angle_deg)
            # Transform to outer hexagon's coordinate frame
            dx = verts[:, 0] - ocx
            dy = verts[:, 1] - ocy
            # Rotate by -outer_angle to align with outer hexagon
            cos_a = np.cos(-oa)
            sin_a = np.sin(-oa)
            rx = dx * cos_a - dy * sin_a
            ry = dx * sin_a + dy * cos_a
            
            # Check against 6 sides of hexagon with flat sides at apothem distance
            # Side normals at angles 30 + k*60 degrees
            for k in range(6):
                theta = np.radians(30 + k * 60)
                val = rx * np.cos(theta) + ry * np.sin(theta)
                if val > max_val:
                    max_val = val
        
        # apothem = S * sqrt(3)/2, so S = max_val / (sqrt(3)/2)
        return max_val / (np.sqrt(3) / 2.0)
=======
    def required_side_length(inner_positions, outer_center=(0.0, 0.0), outer_angle_deg=0.0):
        """Compute minimum side length of outer hexagon to contain all inner hexagons."""
        ocx, ocy = outer_center
        oa = np.radians(outer_angle_deg)
        
        max_val = 0.0
        for (cx, cy, angle_deg) in inner_positions:
            verts = hex_vertices(cx, cy, angle_deg)
            # Transform to outer hexagon's coordinate frame
            dx = verts[:, 0] - ocx
            dy = verts[:, 1] - ocy
            # Rotate by -outer_angle to align with outer hexagon
            cos_a = np.cos(-oa)
            sin_a = np.sin(-oa)
            rx = dx * cos_a - dy * sin_a
            ry = dx * sin_a + dy * cos_a
            
            # Check against 6 sides of hexagon with flat sides at apothem distance
            # Side normals at angles 30 + k*60 degrees
            for k in range(6):
                theta = np.radians(30 + k * 60)
                # val is an array of projections for all 6 vertices
                # We need the maximum projection among all vertices for this side normal
                val_max = np.max(rx * np.cos(theta) + ry * np.sin(theta))
                if val_max > max_val:
                    max_val = val_max
        
        # apothem = S * sqrt(3)/2, so S = max_val / (sqrt(3)/2)
        return max_val / (np.sqrt(3) / 2.0)
>>>>>>> REPLACE

Additionally, there is a potential `UnboundLocalError` for `best_outer_angle` in Strategy 3 if it's never set, but it is set at the end. However, inside Strategy 3, `best_outer_angle` is assigned inside the `if side < best_score` block. If that block never executes, `best_outer_angle` might be undefined if it wasn't defined before. In the original code, `best_outer_angle` is not initialized before the strategies. It is only initialized after Strategy 3. If Strategy 3 doesn't run (e.g., time limit or `best_inner` is None), it falls through to the final block where `best_outer_angle = 0.0` is set. So that is safe.

However, in Strategy 3:
