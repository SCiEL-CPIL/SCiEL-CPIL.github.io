<<<<<<< SEARCH
    def compute_min_outer_side(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        ox = offsets[:, 0]
        oy = offsets[:, 1]
        
        # Fast evaluation: only 6 critical directions per phi, vectorized
        # The outer hexagon side for direction phi is determined by max projection
        # We scan phi in 1-degree steps then refine
        
        best_side = float('inf')
        best_phi = 0.0
        
        # Precompute the 6 normal angle offsets
        normal_offsets = np.pi/6 + np.arange(6) * np.pi/3
        
        # Vectorized scan over phi values
        phi_vals = np.arange(0, 60, 1.0)
        phi_radians = np.radians(phi_vals)
        
        # For each phi, compute all 6 normal directions
        # normal_angles[phi_idx, normal_idx]
        all_normals = phi_radians[:, None] + normal_offsets[None, :]
        nx_all = np.cos(all_normals)  # shape (60, 6)
        ny_all = np.sin(all_normals)  # shape (60, 6)
        
        # proj[phi_idx, vert_idx, normal_idx] = ox[vert_idx]*nx + oy[vert_idx]*ny
        # Use broadcasting: ox[None, :, None] * nx_all[:, None, :] + oy[None, :, None] * ny_all[:, None, :]
        proj = ox[None, :, None] * nx_all[:, None, :] + oy[None, :, None] * ny_all[:, None, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=1)  # shape (60, 6)
        side_vals = (2.0 / sqrt3) * np.max(max_projs, axis=1)  # shape (60,)
        
        best_idx = np.argmin(side_vals)
        best_side = side_vals[best_idx]
        best_phi = phi_vals[best_idx]
        
        # Fine tune with minimize_scalar in a small window
        try:
            lo = max(0.0, best_phi - 1.5)
            hi = min(60.0, best_phi + 1.5)
            
            def eval_phi_fast(phi_deg):
                phi = np.radians(phi_deg)
                normal_angles = phi + normal_offsets
                nx = np.cos(normal_angles)
                ny = np.sin(normal_angles)
                proj = ox[:, None] * nx[None, :] + oy[:, None] * ny[None, :]
                abs_proj = np.abs(proj)
                max_projs = np.max(abs_proj, axis=0)
                side_vals = (2.0 / sqrt3) * max_projs
                return np.max(side_vals)
            
            res = minimize_scalar(eval_phi_fast, bounds=(lo, hi), method='bounded',
                                  options={'xatol': 1e-8, 'maxiter': 30})
            if res.success and res.fun < best_side:
                best_side = res.fun
                best_phi = res.x
        except Exception:
            pass
            
        return best_side, best_phi, center[0], center[1]
=======
    # Precompute normal offsets for outer hexagon
    _normal_offsets = np.pi/6 + np.arange(6) * np.pi/3

    def compute_min_outer_side(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        ox = offsets[:, 0]
        oy = offsets[:, 1]
        nverts = len(ox)

        # Coarse scan: 1-degree steps over [0, 60)
        phi_vals = np.arange(0, 60, 1.0)
        phi_radians = np.radians(phi_vals)
        all_normals = phi_radians[:, None] + _normal_offsets[None, :]
        nx_all = np.cos(all_normals)
        ny_all = np.sin(all_normals)

        # proj shape: (60, nverts, 6)
        proj = ox[None, :, None] * nx_all[:, None, :] + oy[None, :, None] * ny_all[:, None, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=1)
        side_vals = (2.0 / sqrt3) * np.max(max_projs, axis=1)

        best_idx = np.argmin(side_vals)
        best_side = side_vals[best_idx]
        best_phi = phi_vals[best_idx]

        # Fine tune
        try:
            lo = max(0.0, best_phi - 1.5)
            hi = min(60.0, best_phi + 1.5)

            def eval_phi_fast(phi_deg):
                phi = np.radians(phi_deg)
                normal_angles = phi + _normal_offsets
                nx = np.cos(normal_angles)
                ny = np.sin(normal_angles)
                proj = ox[:, None] * nx[None, :] + oy[:, None] * ny[None, :]
                abs_proj = np.abs(proj)
                max_projs = np.max(abs_proj, axis=0)
                side_vals = (2.0 / sqrt3) * max_projs
                return np.max(side_vals)

            res = minimize_scalar(eval_phi_fast, bounds=(lo, hi), method='bounded',
                                  options={'xatol': 1e-8, 'maxiter': 30})
            if res.success and res.fun < best_side:
                best_side = res.fun
                best_phi = res.x
        except Exception:
            pass

        return best_side, best_phi, center[0], center[1]
>>>>>>> REPLACE

