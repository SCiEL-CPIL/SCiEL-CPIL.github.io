<<<<<<< SEARCH
        # 3. Final high-precision minimize_scalar on exact objective
        remaining_time = DEADLINE - (time.time() - t0)
        if remaining_time > 3.0:
            angles_rad = np.radians(best_angles)
            all_verts = get_all_vertices(best_positions, angles_rad)
            verts_flat = all_verts.reshape(-1, 2)
            center = np.mean(verts_flat, axis=0)
            offsets = verts_flat - center

            def eval_phi_final(phi_deg):
                phi = np.radians(phi_deg)
                normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
                nx = np.cos(normal_angles)
                ny = np.sin(normal_angles)
                proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
                abs_proj = np.abs(proj)
                max_projs = np.max(abs_proj, axis=0)
                side_vals = (2.0 / sqrt3) * max_projs
                return np.max(side_vals)

            try:
                lo = max(0.0, best_phi - 2.0)
                hi = min(60.0, best_phi + 2.0)
                res = minimize_scalar(eval_phi_final, bounds=(lo, hi), method='bounded',
                                      options={'xatol': 1e-10, 'maxiter': 200})
                if res.success and res.fun < best_side:
                    best_side = res.fun
                    best_phi = res.x
                    best_cx = center[0]
                    best_cy = center[1]
            except Exception:
                pass
=======
        # 3. Final high-precision minimize_scalar on exact objective
        remaining_time = DEADLINE - (time.time() - t0)
        if remaining_time > 3.0:
            angles_rad = np.radians(best_angles)
            all_verts = get_all_vertices(best_positions, angles_rad)
            verts_flat = all_verts.reshape(-1, 2)
            center = np.mean(verts_flat, axis=0)
            offsets = verts_flat - center

            def eval_phi_final(phi_deg):
                phi = np.radians(phi_deg)
                normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
                nx = np.cos(normal_angles)
                ny = np.sin(normal_angles)
                proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
                abs_proj = np.abs(proj)
                max_projs = np.max(abs_proj, axis=0)
                side_vals = (2.0 / sqrt3) * max_projs
                return np.max(side_vals)

            try:
                lo = max(0.0, best_phi - 1.0)
                hi = min(60.0, best_phi + 1.0)
                res = minimize_scalar(eval_phi_final, bounds=(lo, hi), method='bounded',
                                      options={'xatol': 1e-12, 'maxiter': 500})
                if res.success and res.fun < best_side:
                    best_side = res.fun
                    best_phi = res.x
                    best_cx = center[0]
                    best_cy = center[1]
            except Exception:
                pass
>>>>>>> REPLACE
