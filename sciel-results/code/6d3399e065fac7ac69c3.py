<<<<<<< SEARCH
    # ------------------------------------------------------------------
    # Phase 3: Try alternative outer rotations with L-BFGS-B on inner positions.
    # The outer hexagon rotation is a discrete choice that can affect the
    # optimal inner placement. Try several rotations.
    # ------------------------------------------------------------------
    if time.time() < DEADLINE - 10.0:
        try:
            best_final_S = best_S
            best_final_params = best_centers.copy()
            best_final_outer = best_outer.copy()

            for alpha_test in np.linspace(0, 60, 7):  # 0, 10, 20, ..., 60 degrees
                if time.time() > DEADLINE - 5.0:
                    break

                def obj_alpha(c_flat, alpha=alpha_test):
                    c = c_flat.reshape(n_inner, 2)
                    verts = []
                    for i in range(n_inner):
                        verts.append(_hex_vertices(c[i][0], c[i][1], angles[i]))
                    all_v = np.vstack(verts)
                    outer_center = np.array([best_final_outer[0], best_final_outer[1]])
                    return _min_enclosing_hex_side(all_v, alpha, outer_center)

                x0 = best_final_params.flatten().copy()
                bounds = [(-6, 6), (-6, 6)] * n_inner

                res = minimize(obj_alpha, x0, method='L-BFGS-B',
                               bounds=bounds,
                               options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
                if res.fun < best_final_S:
                    best_final_S = res.fun
                    best_final_params = res.x.reshape(n_inner, 2).copy()
                    best_final_outer = np.array([best_final_outer[0], best_final_outer[1], alpha_test])

            if best_final_S < best_S:
                best_S = best_final_S
                best_centers = best_final_params
                best_outer = best_final_outer
        except Exception:
            pass
=======
    # ------------------------------------------------------------------
    # Phase 3: Try alternative outer rotations with L-BFGS-B on inner positions.
    # The outer hexagon rotation is a discrete choice that can affect the
    # optimal inner placement. Try several rotations.
    # ------------------------------------------------------------------
    if time.time() < DEADLINE - 15.0:
        try:
            best_final_S = best_S
            best_final_params = best_centers.copy()
            best_final_outer = best_outer.copy()

            for alpha_test in [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]:
                if time.time() > DEADLINE - 8.0:
                    break

                def obj_alpha(c_flat, alpha=alpha_test):
                    c = c_flat.reshape(n_inner, 2)
                    verts = []
                    for i in range(n_inner):
                        verts.append(_hex_vertices(c[i][0], c[i][1], angles[i]))
                    all_v = np.vstack(verts)
                    outer_center = np.array([best_final_outer[0], best_final_outer[1]])
                    return _min_enclosing_hex_side(all_v, alpha, outer_center)

                x0 = best_final_params.flatten().copy()
                bounds = [(-6, 6), (-6, 6)] * n_inner

                res = minimize(obj_alpha, x0, method='L-BFGS-B',
                               bounds=bounds,
                               options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
                if res.fun < best_final_S:
                    best_final_S = res.fun
                    best_final_params = res.x.reshape(n_inner, 2).copy()
                    best_final_outer = np.array([best_final_outer[0], best_final_outer[1], alpha_test])

            if best_final_S < best_S:
                best_S = best_final_S
                best_centers = best_final_params
                best_outer = best_final_outer
        except Exception:
            pass
>>>>>>> REPLACE

