<<<<<<< SEARCH
    random.seed(42)
    
    for init_params in configs:
        if time.time() > deadline - 10:
            break
        
        # Try to optimize
        try:
            res = minimize(objective, init_params, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-7})
            
            # Check if result is valid
            centers = [[res.x[i*2], res.x[i*2+1]] for i in range(12)]
            angles = [res.x[24+i] for i in range(12)]
            
            if is_valid(centers, angles):
                val = compute_outer_side(centers, angles, 0.0, 0.0, res.x[36])
                if val < best_val:
                    best_val = val
                    best_params = res.x.copy()
            else:
                # If invalid, try the initial params if they are valid
                if is_valid([[init_params[i*2], init_params[i*2+1]] for i in range(12)], [init_params[24+i] for i in range(12)]):
                    val = compute_outer_side([[init_params[i*2], init_params[i*2+1]] for i in range(12)], [init_params[24+i] for i in range(12)], 0.0, 0.0, init_params[36])
                    if val < best_val:
                        best_val = val
                        best_params = np.array(init_params)
        except Exception:
            pass
=======
    random.seed(42)
    
    for init_params in configs:
        if time.time() > deadline - 20:
            break
        
        # Try to optimize
        try:
            res = minimize(objective, init_params, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8,
                                   'adaptive': True})
            
            # Check if result is valid
            centers = [[res.x[i*2], res.x[i*2+1]] for i in range(12)]
            angles = [res.x[24+i] for i in range(12)]
            
            if is_valid(centers, angles):
                val = compute_outer_side(centers, angles, 0.0, 0.0, res.x[36])
                if val < best_val:
                    best_val = val
                    best_params = res.x.copy()
            else:
                # If invalid, try the initial params if they are valid
                if is_valid([[init_params[i*2], init_params[i*2+1]] for i in range(12)], [init_params[24+i] for i in range(12)]):
                    val = compute_outer_side([[init_params[i*2], init_params[i*2+1]] for i in range(12)], [init_params[24+i] for i in range(12)], 0.0, 0.0, init_params[36])
                    if val < best_val:
                        best_val = val
                        best_params = np.array(init_params)
        except Exception:
            pass

    # Add additional random restarts if time permits
    if time.time() < deadline - 30:
        for restart in range(3):
            if time.time() > deadline - 20:
                break
            try:
                # Generate a perturbed version of the best solution
                if best_params is not None:
                    perturbed = best_params + np.random.randn(37) * 0.1
                else:
                    perturbed = configs[0].copy()
                
                res_r = minimize(objective, perturbed, method='Nelder-Mead',
                               options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8,
                                        'adaptive': True})
                
                centers_r = [[res_r.x[i*2], res_r.x[i*2+1]] for i in range(12)]
                angles_r = [res_r.x[24+i] for i in range(12)]
                
                if is_valid(centers_r, angles_r):
                    val_r = compute_outer_side(centers_r, angles_r, 0.0, 0.0, res_r.x[36])
                    if val_r < best_val:
                        best_val = val_r
                        best_params = res_r.x.copy()
            except Exception:
                pass
>>>>>>> REPLACE

