<<<<<<< SEARCH
    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 10.0:
        cd_side, cd_pos, cd_ang = polish_coordinate_descent(best_positions, best_angles, remaining_time * 0.40)
        if cd_side < best_side:
            best_side = cd_side
            best_positions = cd_pos
            best_angles = cd_ang
            best_side, best_cx, best_cy, best_phi = compute_outer_side_and_center(best_positions, best_angles)
        
        # Phase 3: L-BFGS-B with smooth surrogate from multiple restarts
        remaining_time = DEADLINE - (time.time() - t0)
        if remaining_time > 8.0:
            n_params = N * 3 + 1  # x, y, angle for each hexagon + outer phi
            
            def run_lbfgsb_restart(x0, time_budget):
                """Run L-BFGS-B from a single starting point."""
                t_start = time.time()
                
                def obj(x):
                    if time.time() - t_start > time_budget:
                        return 1e8
                    return smooth_objective(x, beta=30.0, phi_beta=20.0)
                
                try:
                    res = minimize(
                        obj, x0, method='L-BFGS-B',
                        options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 2000}
                    )
                    return res
                except Exception:
                    return None
            
            # Build initial x from current best
            def make_x(pos, ang, phi):
                x = np.zeros(N * 3 + 1)
                for i in range(N):
                    x[i*3] = pos[i, 0]
                    x[i*3+1] = pos[i, 1]
                    x[i*3+2] = ang[i]
                x[N*3] = phi
                return x
            
            x0_base = make_x(best_positions, best_angles, best_phi)
            
            # Try multiple restarts: base, perturbed, and from SA current
            restarts = [x0_base.copy()]
            
            # Perturbed versions
            for r in range(4):
                x_pert = x0_base.copy()
                for i in range(N):
                    x_pert[i*3] += random.gauss(0, 0.3)
                    x_pert[i*3+1] += random.gauss(0, 0.3)
                    x_pert[i*3+2] += random.gauss(0, 3.0)
                x_pert[N*3] += random.gauss(0, 5.0)
                restarts.append(x_pert)
            
            # From SA current state
            x_sa = make_x(current_positions, current_angles, current_phi)
            restarts.append(x_sa)
            
            per_restart_time = remaining_time * 0.55 / len(restarts)
            
            for xi, x_start in enumerate(restarts):
                if time.time() - t0 > DEADLINE - 3.0:
                    break
                
                res = run_lbfgsb_restart(x_start, per_restart_time)
                
                if res is None:
                    continue
                
                x_opt = res.x
                new_pos = np.zeros((N, 2))
                new_ang = np.zeros(N)
                for i in range(N):
                    new_pos[i, 0] = x_opt[i*3]
                    new_pos[i, 1] = x_opt[i*3+1]
                    new_ang[i] = x_opt[i*3+2]
                new_phi = x_opt[N*3]
                
                # Verify validity with exact collision check
                angles_rad = np.radians(new_ang)
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        h1 = (new_pos[i, 0], new_pos[i, 1], angles_rad[i])
                        h2 = (new_pos[j, 0], new_pos[j, 1], angles_rad[j])
                        if check_collision(h1, h2):
                            valid = False
                            break
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                new_side, new_cx, new_cy, new_phi_exact = compute_outer_side_and_center(new_pos, new_ang)
                
                if new_side < best_side:
                    best_side = new_side
                    best_positions = new_pos.copy()
                    best_angles = new_ang.copy()
                    best_cx = new_cx
                    best_cy = new_cy
                    best_phi = new_phi_exact
            
            # Final coordinate descent polish after L-BFGS-B
            remaining_time = DEADLINE - (time.time() - t0)
            if remaining_time > 3.0:
                cd_side2, cd_pos2, cd_ang2 = polish_coordinate_descent(best_positions, best_angles, remaining_time * 0.85)
                if cd_side2 < best_side:
                    best_side = cd_side2
                    best_positions = cd_pos2
                    best_angles = cd_ang2
                    best_side, best_cx, best_cy, best_phi = compute_outer_side_and_center(best_positions, best_angles)
=======
    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 15.0:
        nm_side, nm_pos, nm_ang = polish_nelder_mead(best_positions, best_angles, remaining_time * 0.35)
        if nm_side < best_side:
            best_side = nm_side
            best_positions = nm_pos
            best_angles = nm_ang
            best_side, best_cx, best_cy, best_phi = compute_outer_side_and_center(best_positions, best_angles)
        
        # Phase 3: L-BFGS-B with smooth surrogate from multiple restarts
        remaining_time = DEADLINE - (time.time() - t0)
        if remaining_time > 10.0:
            n_params = N * 3 + 1  # x, y, angle for each hexagon + outer phi
            
            def run_lbfgsb_restart(x0, time_budget):
                """Run L-BFGS-B from a single starting point."""
                t_start = time.time()
                
                def obj(x):
                    if time.time() - t_start > time_budget:
                        return 1e8
                    return smooth_objective(x, beta=30.0, phi_beta=20.0)
                
                try:
                    res = minimize(
                        obj, x0, method='L-BFGS-B',
                        options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 3000}
                    )
                    return res
                except Exception:
                    return None
            
            # Build initial x from current best
            def make_x(pos, ang, phi):
                x = np.zeros(N * 3 + 1)
                for i in range(N):
                    x[i*3] = pos[i, 0]
                    x[i*3+1] = pos[i, 1]
                    x[i*3+2] = ang[i]
                x[N*3] = phi
                return x
            
            x0_base = make_x(best_positions, best_angles, best_phi)
            
            # Try multiple restarts: base, perturbed, and from SA current
            restarts = [x0_base.copy()]
            
            # Perturbed versions with larger perturbations for exploration
            for r in range(6):
                x_pert = x0_base.copy()
                for i in range(N):
                    x_pert[i*3] += random.gauss(0, 0.5)
                    x_pert[i*3+1] += random.gauss(0, 0.5)
                    x_pert[i*3+2] += random.gauss(0, 5.0)
                x_pert[N*3] += random.gauss(0, 8.0)
                restarts.append(x_pert)
            
            # From SA current state
            x_sa = make_x(current_positions, current_angles, current_phi)
            restarts.append(x_sa)
            
            per_restart_time = remaining_time * 0.50 / len(restarts)
            
            for xi, x_start in enumerate(restarts):
                if time.time() - t0 > DEADLINE - 5.0:
                    break
                
                res = run_lbfgsb_restart(x_start, per_restart_time)
                
                if res is None:
                    continue
                
                x_opt = res.x
                new_pos = np.zeros((N, 2))
                new_ang = np.zeros(N)
                for i in range(N):
                    new_pos[i, 0] = x_opt[i*3]
                    new_pos[i, 1] = x_opt[i*3+1]
                    new_ang[i] = x_opt[i*3+2]
                new_phi = x_opt[N*3]
                
                # Verify validity with exact collision check
                angles_rad = np.radians(new_ang)
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        h1 = (new_pos[i, 0], new_pos[i, 1], angles_rad[i])
                        h2 = (new_pos[j, 0], new_pos[j, 1], angles_rad[j])
                        if check_collision(h1, h2):
                            valid = False
                            break
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                new_side, new_cx, new_cy, new_phi_exact = compute_outer_side_and_center(new_pos, new_ang)
                
                if new_side < best_side:
                    best_side = new_side
                    best_positions = new_pos.copy()
                    best_angles = new_ang.copy()
                    best_cx = new_cx
                    best_cy = new_cy
                    best_phi = new_phi_exact
            
            # Final Nelder-Mead polish after L-BFGS-B
            remaining_time = DEADLINE - (time.time() - t0)
            if remaining_time > 5.0:
                nm_side2, nm_pos2, nm_ang2 = polish_nelder_mead(best_positions, best_angles, remaining_time * 0.80)
                if nm_side2 < best_side:
                    best_side = nm_side2
                    best_positions = nm_pos2
                    best_angles = nm_ang2
                    best_side, best_cx, best_cy, best_phi = compute_outer_side_and_center(best_positions, best_angles)
>>>>>>> REPLACE
