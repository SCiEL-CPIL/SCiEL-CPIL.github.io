<<<<<<< SEARCH
    # Polish Phase
    remaining_time = DEADLINE - (time.time() - t0)
    
    if remaining_time > 3.0:
        # 1. Coordinate Descent Polish
        def polish_coordinate_descent(pos, ang, time_limit):
=======
    # Polish Phase
    remaining_time = DEADLINE - (time.time() - t0)
    
    if remaining_time > 3.0:
        # 1. Coordinate Descent Polish
        def compute_smooth_outer_side(positions, angles_deg, beta=50.0):
            """Smooth approximation of min outer side using log-sum-exp."""
            angles_rad = np.radians(angles_deg)
            all_verts = get_all_vertices(positions, angles_rad)
            verts_flat = all_verts.reshape(-1, 2)
            center = np.mean(verts_flat, axis=0)
            offsets = verts_flat - center
            
            def eval_smooth(phi_deg):
                phi = np.radians(phi_deg)
                normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
                nx = np.cos(normal_angles)
                ny = np.sin(normal_angles)
                proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
                abs_proj = np.abs(proj)
                # Soft max via log-sum-exp
                max_projs = np.zeros(6)
                for k in range(6):
                    vals = abs_proj[:, k]
                    m = np.max(vals)
                    max_projs[k] = m + np.log(np.sum(np.exp(vals - m))) / beta
                side_vals = (2.0 / sqrt3) * max_projs
                # Soft max over 6 directions
                m = np.max(side_vals)
                return m + np.log(np.sum(np.exp(side_vals - m))) / beta
            
            # Coarse scan
            phi_vals = np.arange(0, 60, 1.0)
            sides = np.array([eval_smooth(p) for p in phi_vals])
            best_idx = np.argmin(sides)
            best_side = float(sides[best_idx])
            best_phi = float(phi_vals[best_idx])
            
            # Fine tune
            try:
                lo = max(0.0, best_phi - 1.5)
                hi = min(60.0, best_phi + 1.5)
                res = minimize_scalar(eval_smooth, bounds=(lo, hi), method='bounded',
                                      options={'xatol': 1e-10})
                if res.success and res.fun < best_side:
                    best_side = res.fun
                    best_phi = res.x
            except Exception:
                pass
            
            return best_side, best_phi, center[0], center[1]
        
        def smooth_objective(params, beta=50.0):
            """Smooth objective for L-BFGS-B: outer side + collision penalty."""
            pos = np.zeros((N, 2))
            ang = np.zeros(N)
            for i in range(N):
                pos[i, 0] = params[3*i]
                pos[i, 1] = params[3*i+1]
                ang[i] = params[3*i+2]
            
            # Collision penalty: for each pair, compute minimum distance
            angles_rad = np.radians(ang)
            penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    h1 = (pos[i, 0], pos[i, 1], angles_rad[i])
                    h2 = (pos[j, 0], pos[j, 1], angles_rad[j])
                    dx = h1[0] - h2[0]
                    dy = h1[1] - h2[1]
                    dist_sq = dx*dx + dy*dy
                    if dist_sq < 4.5:
                        # Compute actual separation using separating axis
                        c1, s1 = math.cos(h1[2]), math.sin(h1[2])
                        c2, s2 = math.cos(h2[2]), math.sin(h2[2])
                        v1x = h1[0] + _ux * c1 - _uy * s1
                        v1y = h1[1] + _ux * s1 + _uy * c1
                        v2x = h2[0] + _ux * c2 - _uy * s2
                        v2y = h2[1] + _ux * s2 + _uy * c2
                        en1x = _enx * c1 - _eny * s1
                        en1y = _enx * s1 + _eny * c1
                        en2x = _enx * c2 - _eny * s2
                        en2y = _enx * s2 + _eny * c2
                        min_sep = float('inf')
                        for i_e in range(6):
                            nx, ny = en1x[i_e], en1y[i_e]
                            p1 = v1x * nx + v1y * ny
                            p2 = v2x * nx + v2y * ny
                            sep = min(p1.max() - p2.min(), p2.max() - p1.min())
                            if sep < 0:
                                min_sep = min(min_sep, -sep)
                        for i_e in range(6):
                            nx, ny = en2x[i_e], en2y[i_e]
                            p1 = v1x * nx + v1y * ny
                            p2 = v2x * nx + v2y * ny
                            sep = min(p1.max() - p2.min(), p2.max() - p1.min())
                            if sep < 0:
                                min_sep = min(min_sep, -sep)
                        if min_sep < 0:
                            penalty += min_sep * min_sep * 100.0
                        elif min_sep < 0.01:
                            penalty += (0.01 - min_sep) * 10.0
                    else:
                        # Near-miss penalty for smoothness
                        if dist_sq < 6.0:
                            penalty += (6.0 - dist_sq) * 0.5
            
            side, _, _, _ = compute_smooth_outer_side(pos, ang, beta)
            return side + penalty
        
        def polish_coordinate_descent(pos, ang, time_limit):
>>>>>>> REPLACE

