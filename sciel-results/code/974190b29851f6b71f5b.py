    def calc_S(alpha_beta):
        alpha, beta = alpha_beta
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf')
        
        # Rotate inner centers by alpha
        cos_a = math.cos(alpha)
        sin_a = math.sin(alpha)
        rot_pts = np.column_stack([
            rel_pts[:, 0] * cos_a - rel_pts[:, 1] * sin_a,
            rel_pts[:, 0] * sin_a + rel_pts[:, 1] * cos_a
        ])
        
        max_S = 0.0
        # Outer hexagon edge normals
        # The outer hexagon is rotated by beta.
        # Its edge normals are at beta + k*pi/3 + pi/6?
        # Standard hexagon with flat top: edges are horizontal. Normals are vertical (pi/2).
        # If we define the hexagon by its side length S and rotation beta,
        # we need to be consistent.
        # Let's assume the "rotation" beta rotates the hexagon from a standard orientation.
        # Standard orientation: Vertices at 0, 60, ...? Or Edges at 0, 60...?
        # The code uses: ang = beta + math.pi / 6.0 + k * math.pi / 3.0
        # This generates 6 angles 60 deg apart.
        # If beta=0, angles are 30, 90, 150, 210, 270, 330.
        # These are the normals to the edges of a hexagon with a vertex at 0 deg?
        # If a vertex is at 0 deg, the edge between -30 and 30 has normal 0? No.
        # Let's stick to the code's definition of normals, but ensure the support function logic matches.
        
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            nx = math.cos(ang)
            ny = math.sin(ang)
            
            # Calculate max projection of inner hexagons onto this normal
            max_proj_val = 0.0
            for i in range(N):
                cx = rot_pts[i, 0]
                cy = rot_pts[i, 1]
                # Center projection
                p_center = cx * nx + cy * ny
                
                # Support function of inner hexagon in direction ang
                # Inner hexagon is rotated by alpha.
                # Relative angle
                rel_ang = ang - alpha
                # Normalize to [0, pi/3)
                rel_ang_mod = rel_ang % (math.pi / 3.0)
                # Distance to nearest vertex (which are at multiples of pi/3)
                # The vertex angles are 0, pi/3, ... relative to alpha.
                # So relative to alpha, the normal is at rel_ang.
                # The closest vertex is at 0 or pi/3.
                # Distance d = min(rel_ang_mod, pi/3 - rel_ang_mod)
                d = min(rel_ang_mod, math.pi / 3.0 - rel_ang_mod)
                # Support is cos(d) because circumradius is 1.
                r_inner = math.cos(d)
                
                p_total = p_center + r_inner
                if abs(p_total) > max_proj_val:
                    max_proj_val = abs(p_total)
            
            required_apothem = max_proj_val
            required_S = required_apothem / APOTHEM
            if required_S > max_S:
                max_S = required_S
                
        return max_S
    