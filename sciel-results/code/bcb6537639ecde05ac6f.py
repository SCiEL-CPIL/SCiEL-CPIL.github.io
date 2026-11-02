    def get_best_config(positions):
        # Calculate penalty once
        diffs = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        projs_diff = np.einsum('ijk,lk->ijl', diffs, NORMALS)
        abs_projs = np.abs(projs_diff)
        max_projs = np.max(abs_projs, axis=2)
        mask = np.ones((12, 12), dtype=bool)
        np.fill_diagonal(mask, False)
        overlaps = (max_projs < INRADIUS_EXCL) & mask
        depths = np.where(overlaps, INRADIUS_EXCL - max_projs, 0)
        penalty = 10.0 * np.sum(depths)
        
        best_L = float('inf')
        best_theta = 0.0
        for theta in np.linspace(0, 30, 31):
            # Calculate L_outer only
            verts = get_vertices(positions)
            angles = np.deg2rad(theta + 30 + np.arange(6) * 60)
            normals = np.column_stack([np.cos(angles), np.sin(angles)])
            projs = verts @ normals.T
            M = np.max(projs, axis=0)
            m = np.min(projs, axis=0)
            C_k = (M + m) / 2
            Delta_k = (M - m) / 2
            center = np.mean(verts, axis=0)
            u_k = center @ normals.T
            d_k = np.abs(C_k - u_k) + Delta_k
            d = np.max(d_k)
            L_outer = 2 * d / np.sqrt(3)
            
            L = L_outer + penalty
            if L < best_L:
                best_L = L
                best_theta = theta
        
        # Refine
        if best_theta > 0 and best_theta < 30:
             for theta in np.linspace(best_theta - 0.5, best_theta + 0.5, 11):
                 if 0 <= theta <= 30:
                     verts = get_vertices(positions)
                     angles = np.deg2rad(theta + 30 + np.arange(6) * 60)
                     normals = np.column_stack([np.cos(angles), np.sin(angles)])
                     projs = verts @ normals.T
                     M = np.max(projs, axis=0)
                     m = np.min(projs, axis=0)
                     C_k = (M + m) / 2
                     Delta_k = (M - m) / 2
                     center = np.mean(verts, axis=0)
                     u_k = center @ normals.T
                     d_k = np.abs(C_k - u_k) + Delta_k
                     d = np.max(d_k)
                     L_outer = 2 * d / np.sqrt(3)
                     L = L_outer + penalty
                     if L < best_L:
                         best_L = L
                         best_theta = theta
        return best_L, best_theta
