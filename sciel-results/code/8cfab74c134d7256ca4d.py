<<<<<<< SEARCH
    # Phase 4: High-precision polish - fine-tune inner positions and outer hex
    if time.time() < DEADLINE - 10.0:
        # Try small perturbations of inner positions to see if we can squeeze more
        for n_inner in [len(angles)]:
            x0_fine = np.zeros(3 + 2 * n_inner)
            x0_fine[0] = best_vars_outer[0]
            x0_fine[1] = best_vars_outer[1]
            x0_fine[2] = best_vars_outer[2]
            for i in range(n_inner):
                x0_fine[3 + 2*i] = best_centers[i][0]
                x0_fine[4 + 2*i] = best_centers[i][1]
            
            try:
                res = minimize(_joint_objective, x0_fine, args=(angles,), 
                              method='L-BFGS-B', 
                              options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12})
                
                if time.time() < DEADLINE - 5.0:
                    outer_cx, outer_cy, outer_angle = res.x[0], res.x[1], res.x[2]
                    new_centers = np.array([[res.x[3+2*i], res.x[4+2*i]] for i in range(n_inner)])
                    S_trial = _min_enclosing_hex_side_fast(new_centers, angles, outer_angle, np.array([outer_cx, outer_cy]))
                    pen_trial = _overlap_penalty(new_centers, angles)
                    
                    if pen_trial < 1e-10 and S_trial < best_S:
                        best_S = S_trial
                        best_vars_outer = np.array([outer_cx, outer_cy, outer_angle])
                        best_centers = new_centers
            except Exception:
                pass
=======
    # Phase 4: High-precision polish - fine-tune inner positions and outer hex
    if time.time() < DEADLINE - 10.0:
        # Try small perturbations of inner positions to see if we can squeeze more
        n_inner = len(angles)
        x0_fine = np.zeros(3 + 2 * n_inner)
        x0_fine[0] = best_vars_outer[0]
        x0_fine[1] = best_vars_outer[1]
        x0_fine[2] = best_vars_outer[2]
        for i in range(n_inner):
            x0_fine[3 + 2*i] = best_centers[i][0]
            x0_fine[4 + 2*i] = best_centers[i][1]
        
        # Try multiple beta values for final polish
        for beta in [100.0, 200.0]:
            if time.time() > DEADLINE - 5.0:
                break
            
            try:
                res = minimize(_joint_objective_smooth, x0_fine, args=(angles, beta), 
                              method='L-BFGS-B', 
                              options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12})
                
                if time.time() < DEADLINE - 3.0:
                    outer_cx, outer_cy, outer_angle = res.x[0], res.x[1], res.x[2]
                    new_centers = np.array([[res.x[3+2*i], res.x[4+2*i]] for i in range(n_inner)])
                    S_trial = _min_enclosing_hex_side_fast(new_centers, angles, outer_angle, np.array([outer_cx, outer_cy]))
                    pen_trial = _overlap_penalty(new_centers, angles)
                    
                    if pen_trial < 1e-10 and S_trial < best_S:
                        best_S = S_trial
                        best_vars_outer = np.array([outer_cx, outer_cy, outer_angle])
                        best_centers = new_centers
            except Exception:
                pass
>>>>>>> REPLACE

