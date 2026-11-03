<<<<<<< SEARCH
            init_params = np.zeros(3*N)
            for i in range(N):
                init_params[3*i] = best_positions[i, 0]
                init_params[3*i+1] = best_positions[i, 1]
                init_params[3*i+2] = best_angles[i]
                
            try:
                res = minimize(polish_objective, init_params, method='Nelder-Mead', 
                               options={'maxiter': 20000, 'xatol': 1e-9, 'fatol': 1e-11,
                                        'adaptive': True})
                
                if res.fun < best_side:
                    new_pos = np.zeros((N, 2))
                    new_ang = np.zeros(N)
                    for i in range(N):
                        new_pos[i, 0] = res.x[3*i]
                        new_pos[i, 1] = res.x[3*i+1]
                        new_ang[i] = res.x[3*i+2]
                    
                    if is_valid_configuration(new_pos, new_ang):
                        new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_pos, new_ang)
                        if new_side < best_side:
                            best_side = new_side
                            best_positions = new_pos
                            best_angles = new_ang
                            best_phi = new_phi
                            best_cx = new_cx
                            best_cy = new_cy
            except Exception:
                pass
=======
            init_params = np.zeros(3*N)
            for i in range(N):
                init_params[3*i] = best_positions[i, 0]
                init_params[3*i+1] = best_positions[i, 1]
                init_params[3*i+2] = best_angles[i]
                
            try:
                res = minimize(polish_objective, init_params, method='Nelder-Mead', 
                               options={'maxiter': 30000, 'xatol': 1e-10, 'fatol': 1e-12,
                                        'adaptive': True})
                
                if res.fun < best_side:
                    new_pos = np.zeros((N, 2))
                    new_ang = np.zeros(N)
                    for i in range(N):
                        new_pos[i, 0] = res.x[3*i]
                        new_pos[i, 1] = res.x[3*i+1]
                        new_ang[i] = res.x[3*i+2]
                    
                    if is_valid_configuration(new_pos, new_ang):
                        new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_pos, new_ang)
                        if new_side < best_side:
                            best_side = new_side
                            best_positions = new_pos
                            best_angles = new_ang
                            best_phi = new_phi
                            best_cx = new_cx
                            best_cy = new_cy
            except Exception:
                pass
            
            # Additional: try multiple Nelder-Mead runs from slightly perturbed starts
            remaining_time2 = DEADLINE - (time.time() - t0)
            if remaining_time2 > 5.0 and best_side > 3.0:
                for attempt in range(3):
                    if time.time() - t0 > DEADLINE - 2.0:
                        break
                    try:
                        perturbed = init_params.copy()
                        for i in range(3*N):
                            perturbed[i] += random.gauss(0, 0.02)
                        res2 = minimize(polish_objective, perturbed, method='Nelder-Mead',
                                       options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12,
                                                'adaptive': True})
                        if res2.fun < best_side:
                            new_pos = np.zeros((N, 2))
                            new_ang = np.zeros(N)
                            for i in range(N):
                                new_pos[i, 0] = res2.x[3*i]
                                new_pos[i, 1] = res2.x[3*i+1]
                                new_ang[i] = res2.x[3*i+2]
                            if is_valid_configuration(new_pos, new_ang):
                                new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_pos, new_ang)
                                if new_side < best_side:
                                    best_side = new_side
                                    best_positions = new_pos
                                    best_angles = new_ang
                                    best_phi = new_phi
                                    best_cx = new_cx
                                    best_cy = new_cy
                    except Exception:
                        pass
>>>>>>> REPLACE
