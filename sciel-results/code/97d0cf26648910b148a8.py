<<<<<<< SEARCH
    # Now optimize inner hexagon rotations independently
    # For each inner hex, try rotating it to minimize the required S
    # We'll do a local search on each inner hex angle
    
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        # Optimize each inner hex angle one at a time (coordinate descent)
        inner_angles = [best_beta] * n_pts
        
        # Multiple passes of coordinate descent
        for pass_num in range(3):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            for i in range(n_pts):
                if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                    break
                
                def obj_angle_i(theta_i):
                    angles = inner_angles[:]
                    angles[i] = theta_i
                    _, _, S = solve_lp(best_beta, angles)
                    return S
                
                # Try a few candidate angles around current best
                current_angle = inner_angles[i]
                best_angle_i = current_angle
                best_S_i = obj_angle_i(current_angle)
                
                # Try angles in [current - pi/6, current + pi/6] with some candidates
                candidates = []
                for delta in np.linspace(-math.pi/6, math.pi/6, 7):
                    candidates.append(current_angle + delta)
                # Also try 0, pi/6, pi/3, etc.
                for base in [0, math.pi/6, math.pi/3, math.pi/2, 2*math.pi/3, 5*math.pi/6]:
                    candidates.append(base)
                
                for cand in candidates:
                    S_cand = obj_angle_i(cand)
                    if S_cand < best_S_i:
                        best_S_i = S_cand
                        best_angle_i = cand
                
                if best_S_i < best_S - 1e-12:
                    inner_angles[i] = best_angle_i
                    best_S = best_S_i
                    best_cx, best_cy, _ = solve_lp(best_beta, inner_angles)
        
        # Final polish: optimize beta again with optimized inner angles
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            def obj_beta_final(beta):
                _, _, S = solve_lp(beta, inner_angles)
                return S
            
            res = minimize_scalar(obj_beta_final, bounds=(0, math.pi / 6), method='bounded',
                                 options={'maxiter': 500, 'xatol': 1e-10})
            if res.fun < best_S:
                best_beta = res.x
                best_cx, best_cy, best_S = solve_lp(best_beta, inner_angles)
                
                # Polish inner angles again
                for i in range(n_pts):
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    def obj_angle_i(theta_i):
                        angles = inner_angles[:]
                        angles[i] = theta_i
                        _, _, S = solve_lp(best_beta, angles)
                        return S
                    
                    current_angle = inner_angles[i]
                    best_angle_i = current_angle
                    best_S_i = obj_angle_i(current_angle)
                    
                    candidates = np.linspace(current_angle - math.pi/12, current_angle + math.pi/12, 13)
                    for cand in candidates:
                        S_cand = obj_angle_i(cand)
                        if S_cand < best_S_i:
                            best_S_i = S_cand
                            best_angle_i = cand
                    
                    if best_S_i < best_S - 1e-12:
                        inner_angles[i] = best_angle_i
                        best_S = best_S_i
                        best_cx, best_cy, _ = solve_lp(best_beta, inner_angles)
=======
    # Now optimize inner hexagon rotations independently
    # For each inner hex, try rotating it to minimize the required S
    # We'll do a local search on each inner hex angle
    
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        # Optimize each inner hex angle one at a time (coordinate descent)
        inner_angles = [best_beta] * n_pts
        
        # Multiple passes of coordinate descent with progressively finer grids
        n_passes = 5
        for pass_num in range(n_passes):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            # Grid size decreases with each pass for finer search
            n_candidates = max(9, 25 - 4 * pass_num)
            angle_range = math.pi / 6.0 / (2 ** pass_num)
            
            for i in range(n_pts):
                if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                    break
                
                def obj_angle_i(theta_i):
                    angles = inner_angles[:]
                    angles[i] = theta_i
                    _, _, S = solve_lp(best_beta, angles)
                    return S
                
                # Try candidate angles around current best
                current_angle = inner_angles[i]
                best_angle_i = current_angle
                best_S_i = obj_angle_i(current_angle)
                
                # Fine grid around current angle
                candidates = np.linspace(current_angle - angle_range, current_angle + angle_range, n_candidates)
                # Also try symmetric angles relative to beta
                for delta in np.linspace(-math.pi/6, math.pi/6, 7):
                    candidates.append(best_beta + delta)
                
                for cand in candidates:
                    S_cand = obj_angle_i(cand)
                    if S_cand < best_S_i:
                        best_S_i = S_cand
                        best_angle_i = cand
                
                if best_S_i < best_S - 1e-14:
                    inner_angles[i] = best_angle_i
                    best_S = best_S_i
                    best_cx, best_cy, _ = solve_lp(best_beta, inner_angles)
        
        # Final polish: optimize beta again with optimized inner angles
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            def obj_beta_final(beta):
                _, _, S = solve_lp(beta, inner_angles)
                return S
            
            # Multi-start for beta optimization
            beta_starts = [0.0, math.pi/12, math.pi/6/2, math.pi/6*0.8, math.pi/6*0.2]
            for b_start in beta_starts:
                if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                    break
                try:
                    res = minimize_scalar(obj_beta_final, bounds=(0, math.pi / 6), method='bounded',
                                         options={'maxiter': 1000, 'xatol': 1e-12, 'x0': b_start})
                    if res.fun < best_S:
                        best_beta = res.x
                        best_cx, best_cy, best_S = solve_lp(best_beta, inner_angles)
                except Exception:
                    pass
            
            # Polish inner angles again with very fine grid
            for i in range(n_pts):
                if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                    break
                def obj_angle_i(theta_i):
                    angles = inner_angles[:]
                    angles[i] = theta_i
                    _, _, S = solve_lp(best_beta, angles)
                    return S
                
                current_angle = inner_angles[i]
                best_angle_i = current_angle
                best_S_i = obj_angle_i(current_angle)
                
                # Very fine grid for final polish
                candidates = np.linspace(current_angle - math.pi/30, current_angle + math.pi/30, 21)
                for cand in candidates:
                    S_cand = obj_angle_i(cand)
                    if S_cand < best_S_i:
                        best_S_i = S_cand
                        best_angle_i = cand
                
                if best_S_i < best_S - 1e-14:
                    inner_angles[i] = best_angle_i
                    best_S = best_S_i
                    best_cx, best_cy, _ = solve_lp(best_beta, inner_angles)
>>>>>>> REPLACE

