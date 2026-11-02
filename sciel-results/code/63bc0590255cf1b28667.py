<<<<<<< SEARCH
    # --- Optimization ---
    
    # Bounds for the 2 base hexagons
    # x, y in [-4, 4]
    # angle in [0, pi/3]
    bounds = [
        (-4.0, 4.0),  # x1
        (-4.0, 4.0),  # y1
        (0.0, np.pi/3), # a1
        (-4.0, 4.0),  # x2
        (-4.0, 4.0),  # y2
        (0.0, np.pi/3), # a2
    ]
    
    best_s = np.inf
    best_params = None
    
    # Use Differential Evolution for global search
    try:
        res = differential_evolution(
            compute_objective_symmetric,
            bounds,
            maxiter=200,
            popsize=15,
            tol=1e-5,
            seed=42,
            polish=True # Use L-BFGS-B to polish the result
        )
        
        if res.fun < best_s:
            best_s = res.fun
            best_params = res.x
            
    except Exception as e:
        pass
    
    # Fallback or Refinement: Try a few local minimizations from good symmetric starts
    if best_params is not None:
        # Polish with L-BFGS-B
        try:
            res_local = minimize(
                compute_objective_symmetric,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12}
            )
            if res_local.fun < best_s:
                best_s = res_local.fun
                best_params = res_local.x
        except Exception:
            pass
    
    # Construct the final answer from best_params
    if best_params is None:
        # Fallback: Simple grid or known approximate
        # Try a few random symmetric starts
        for _ in range(5):
            p = np.array([
                np.random.uniform(1, 3), np.random.uniform(-0.5, 0.5), np.random.uniform(0, np.pi/3),
                np.random.uniform(2, 4), np.random.uniform(-0.5, 0.5), np.random.uniform(0, np.pi/3)
            ])
            val = compute_objective_symmetric(p)
            if val < best_s:
                best_s = val
                best_params = p
            try:
                res_local = minimize(
                    compute_objective_symmetric,
                    p,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 50}
                )
                if res_local.fun < best_s:
                    best_s = res_local.fun
                    best_params = res_local.x
            except Exception:
                pass
            if time.time() > deadline - 5:
                break
=======
    # --- Optimization ---
    
    # Bounds for the 2 base hexagons
    # x, y in [-4, 4]
    # angle in [0, pi/3]
    bounds = [
        (-4.0, 4.0),  # x1
        (-4.0, 4.0),  # y1
        (0.0, np.pi/3), # a1
        (-4.0, 4.0),  # x2
        (-4.0, 4.0),  # y2
        (0.0, np.pi/3), # a2
    ]
    
    best_s = np.inf
    best_params = None
    
    # Use Differential Evolution for global search with more iterations and larger population
    try:
        res = differential_evolution(
            compute_objective_symmetric,
            bounds,
            maxiter=300,
            popsize=20,
            tol=1e-6,
            seed=42,
            polish=True,
            mutation=(0.4, 1.0),
            recombination=0.8
        )
        
        if res.fun < best_s:
            best_s = res.fun
            best_params = res.x
            
    except Exception as e:
        pass
    
    # Additional local minimizations from multiple starting points
    if best_params is not None:
        # Polish with L-BFGS-B using tighter tolerances
        try:
            res_local = minimize(
                compute_objective_symmetric,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-14, 'maxls': 50}
            )
            if res_local.fun < best_s:
                best_s = res_local.fun
                best_params = res_local.x
        except Exception:
            pass
    
    # Try additional random restarts if time permits
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        n_restarts = min(8, int(remaining_time / 10))
        for restart_idx in range(n_restarts):
            if time.time() > deadline - 5:
                break
            # Generate diverse starting points
            p = np.array([
                np.random.uniform(0.5, 3.5), np.random.uniform(-1.0, 1.0), np.random.uniform(0, np.pi/3),
                np.random.uniform(1.5, 4.0), np.random.uniform(-1.0, 1.0), np.random.uniform(0, np.pi/3)
            ])
            val = compute_objective_symmetric(p)
            if val < best_s:
                best_s = val
                best_params = p
            try:
                res_local = minimize(
                    compute_objective_symmetric,
                    p,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-13, 'gtol': 1e-13}
                )
                if res_local.fun < best_s:
                    best_s = res_local.fun
                    best_params = res_local.x
            except Exception:
                pass
>>>>>>> REPLACE

