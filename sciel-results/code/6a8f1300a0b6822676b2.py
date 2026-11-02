<<<<<<< SEARCH
    # Compute proper center for the best configuration
    if best_inner is not None:
        all_vertices = np.vstack([_hex_vertices(p[0], p[1], p[2]) for p in best_inner])
        # Find center via LP
        base_angle = math.radians(best_outer_angle + 30)
        angles = base_angle + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        
        M = len(all_vertices)
        A_rows = []
        b_vals = []
        for i in range(M):
            vi = all_vertices[i]
            for k in range(6):
                nk = normals[k]
                A_rows.append([-nk[0], -nk[1], -1.0])
                b_vals.append(-nk[0]*vi[0] - nk[1]*vi[1])
                A_rows.append([nk[0], nk[1], -1.0])
                b_vals.append(nk[0]*vi[0] + nk[1]*vi[1])
        
        A = np.array(A_rows)
        b = np.array(b_vals)
        c_obj = np.array([0.0, 0.0, 1.0])
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')
            if res.status == 0:
                best_center = (res.x[0], res.x[1])
                best_side = res.x[2] * 2.0 / _SQRT3
        except Exception:
            pass
=======
    # Compute proper center for the best configuration
    if best_inner is not None:
        all_vertices = np.vstack([_hex_vertices(p[0], p[1], p[2]) for p in best_inner])
        # Find center via LP
        base_angle = math.radians(best_outer_angle + 30)
        angles = base_angle + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        
        M = len(all_vertices)
        A_rows = []
        b_vals = []
        for i in range(M):
            vi = all_vertices[i]
            for k in range(6):
                nk = normals[k]
                A_rows.append([-nk[0], -nk[1], -1.0])
                b_vals.append(-nk[0]*vi[0] - nk[1]*vi[1])
                A_rows.append([nk[0], nk[1], -1.0])
                b_vals.append(nk[0]*vi[0] + nk[1]*vi[1])
        
        A = np.array(A_rows)
        b = np.array(b_vals)
        c_obj = np.array([0.0, 0.0, 1.0])
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')
            if res.status == 0:
                best_center = (res.x[0], res.x[1])
                best_side = res.x[2] * 2.0 / _SQRT3
        except Exception:
            pass
        
        # Final high-precision polish: tighten the LP solution
        if time.time() < _DEADLINE - 5.0 and best_inner is not None:
            # Use the LP center as starting point and do a final Nelder-Mead pass
            x0 = np.array([p[0] for p in best_inner] + 
                          [p[1] for p in best_inner] + [best_outer_angle])
            
            def final_obj(params):
                n_pos = N * 2
                positions = [(params[i], params[i+1]) for i in range(0, n_pos, 2)]
                oa = params[-1]
                all_v = np.vstack([_hex_vertices(px, py, p[2]) for (px, py), p in zip(positions, best_inner)])
                side = _enclosing_hex_side(all_v, oa)
                penalty = 0.0
                for i in range(N):
                    for j in range(i+1, N):
                        di = np.array([positions[j][0]-positions[i][0], positions[j][1]-positions[i][1]])
                        d = _hex_distance_fast(di, best_inner[i][2], best_inner[j][2])
                        if d < 0:
                            penalty += d*d*100.0
                return side + penalty
            
            try:
                res2 = minimize(final_obj, x0, method='Nelder-Mead',
                               options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10})
                if res2.fun < best_side:
                    best_side = res2.fun
                    for i in range(N):
                        best_inner[i][0] = res2.x[i]
                        best_inner[i][1] = res2.x[N+i]
                    best_outer_angle = res2.x[-1]
                    # Recompute center
                    all_v2 = np.vstack([_hex_vertices(p[0], p[1], p[2]) for p in best_inner])
                    try:
                        res3 = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')
                        if res3.status == 0:
                            best_center = (res3.x[0], res3.x[1])
                            best_side = res3.x[2] * 2.0 / _SQRT3
                    except Exception:
                        pass
            except Exception:
                pass
>>>>>>> REPLACE
