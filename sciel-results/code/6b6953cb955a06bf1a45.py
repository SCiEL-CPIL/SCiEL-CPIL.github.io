<<<<<<< SEARCH
    # 3. Try auxiliary variable formulation
    if time.time() < deadline - 15:
        t_init = current_score
        init_params = np.append(current_pts.flatten(), t_init)
        objective_t = create_objective_with_t(current_pts, 100.0)
        
        try:
            res = minimize(objective_t, init_params, method='L-BFGS-B',
                           bounds=[(-1.0, 1.0)] * (2 * N) + [(0.0, 1.0)],
                           options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
            
            if res.success or res.fun < 0:
                new_pts = res.x[:-1].reshape(-1, 2)
                new_score = compute_exact_score(new_pts)
                
                if new_score > current_score:
                    current_pts = new_pts
                    current_score = new_score
        except Exception:
            pass

    # 4. Local polish with high beta and tight tolerances
    if time.time() < deadline - 5:
        objective = create_objective(current_pts, 5000.0)
        try:
            res = minimize(objective, current_pts.flatten(), method='L-BFGS-B',
                           bounds=[(-1.0, 1.0)] * (2 * N),
                           options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12})
            
            if res.success or res.fun < 0:
                new_pts = res.x.reshape(-1, 2)
                new_score = compute_exact_score(new_pts)
                
                if new_score > current_score:
                    current_pts = new_pts
                    current_score = new_score
        except Exception:
            pass
=======
    # 3. Try auxiliary variable formulation
    if time.time() < deadline - 20:
        t_init = current_score
        init_params = np.append(current_pts.flatten(), t_init)
        objective_t = create_objective_with_t(current_pts, 200.0)
        
        try:
            res = minimize(objective_t, init_params, method='L-BFGS-B',
                           bounds=[(-1.0, 1.0)] * (2 * N) + [(0.0, 1.0)],
                           options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
            
            if res.success or res.fun < 0:
                new_pts = res.x[:-1].reshape(-1, 2)
                new_score = compute_exact_score(new_pts)
                
                if new_score > current_score:
                    current_pts = new_pts
                    current_score = new_score
        except Exception:
            pass

    # 4. Final force-directed polish
    if time.time() < deadline - 8:
        polished = force_directed_refine(current_pts, iterations=150, dt=0.0005)
        polished_score = compute_exact_score(polished)
        if polished_score > current_score:
            current_pts = polished
            current_score = polished_score

    # 5. Local polish with high beta and tight tolerances
    if time.time() < deadline - 5:
        objective = create_objective(current_pts, 5000.0)
        try:
            res = minimize(objective, current_pts.flatten(), method='L-BFGS-B',
                           bounds=[(-1.0, 1.0)] * (2 * N),
                           options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12})
            
            if res.success or res.fun < 0:
                new_pts = res.x.reshape(-1, 2)
                new_score = compute_exact_score(new_pts)
                
                if new_score > current_score:
                    current_pts = new_pts
                    current_score = new_score
        except Exception:
            pass

    # 6. Final force-directed micro-refinement
    if time.time() < deadline - 2:
        micro_polished = force_directed_refine(current_pts, iterations=100, dt=0.0001)
        micro_score = compute_exact_score(micro_polished)
        if micro_score > current_score:
            current_pts = micro_polished
            current_score = micro_score
>>>>>>> REPLACE

