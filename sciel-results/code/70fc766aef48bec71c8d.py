    def objective(params):
        pos_flat = params[:2*N]
        angles = params[2*N:3*N]
        theta_out = params[3*N]
        
        positions = pos_flat.reshape(N, 2)
        
        side = calc_min_outer_side(positions, angles, theta_out)
        
        penalty = 0.0
        verts = get_all_verts(positions, angles)
        
        # Precompute normals for all hexagons
        # normals[i] = unit_edge_normals @ R(angles[i]).T
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)
        # R = [[c, -s], [s, c]]
        # unit_edge_normals is (6, 2)
        # We want normals[i] (6, 2)
        # normals[i][k] = unit_edge_normals[k] @ R[angle_i].T
        # = [u_kx * c + u_ky * s, -u_kx * s + u_ky * c]
        
        uex = unit_edge_normals[:, 0]
        uey = unit_edge_normals[:, 1]
        
        # (N, 6)
        nx = uex[None, :] * cos_a[:, None] + uey[None, :] * sin_a[:, None]
        ny = -uex[None, :] * sin_a[:, None] + uey[None, :] * cos_a[:, None]
        
        all_normals = np.stack([nx, ny], axis=-1) # (N, 6, 2)
        
        for i in range(N):
            for j in range(i+1, N):
                gap = sat_gap_optimized(verts[i], verts[j], all_normals[i], all_normals[j])
                if gap < 0:
                    penalty += gap * gap
        
        return side + 10.0 * penalty
