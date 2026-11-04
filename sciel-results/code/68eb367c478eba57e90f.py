<<<<<<< SEARCH
        return np.column_stack((xs_best, ys_best, rs_best))
    
    # Try diverse aspect ratios with multiple initialization strategies
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.2, 0.7, 1.3, 0.6, 1.4]
    
    for w_init in aspect_ratios:
        if time.time() > deadline:
            break
            
        for strategy in range(3):
            if time.time() > deadline:
                break
                
            xs, ys, rs = generate_grid_init(w_init, strategy)
            x0_curr = np.concatenate(([w_init], xs, ys, rs))
            
            # Use L-BFGS-B with tighter tolerances
            try:
                res = minimize(objective, x0_curr, method='L-BFGS-B', 
                             options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50})
                
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x.copy()
                    
                    # Second pass with tighter tolerance from improved starting point
                    if time.time() < deadline - 15:
                        res2 = minimize(objective, best_x, method='L-BFGS-B', 
                                     options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14})
                        if res2.fun < best_val:
                            best_val = res2.fun
                            best_x = res2.x.copy()
            except:
                pass
    
    if best_x is None:
=======
        return np.column_stack((xs_best, ys_best, rs_best))
    
    # Try diverse aspect ratios with multiple initialization strategies
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.2, 0.7, 1.3, 0.6, 1.4]
    
    # Prepare constraints and bounds for SLSQP
    constraints = make_constraints()
    bounds = [(0.1, 1.9)] + [(0, 2)] * N + [(0, 2)] * N + [(0, 1)] * N

    for w_init in aspect_ratios:
        if time.time() > deadline:
            break
            
        for strategy in range(3):
            if time.time() > deadline:
                break
                
            xs, ys, rs = generate_grid_init(w_init, strategy)
            x0_curr = np.concatenate(([w_init], xs, ys, rs))
            
            # Use SLSQP with explicit constraints
            try:
                res = minimize(objective, x0_curr, method='SLSQP', 
                             bounds=bounds,
                             constraints=constraints,
                             options={'maxiter': 200, 'ftol': 1e-9})
                
                if res.success and res.fun < best_val:
                    # Verify validity manually before accepting
                    w_val = res.x[0]
                    h_val = 2.0 - w_val
                    xs_val = res.x[1:N+1]
                    ys_val = res.x[N+1:2*N+1]
                    rs_val = res.x[2*N+1:3*N+1]
                    
                    valid = True
                    for i in range(N):
                        if xs_val[i] < rs_val[i] - 1e-6 or xs_val[i] > w_val - rs_val[i] + 1e-6:
                            valid = False
                        if ys_val[i] < rs_val[i] - 1e-6 or ys_val[i] > h_val - rs_val[i] + 1e-6:
                            valid = False
                    for i in range(N):
                        for j in range(i+1, N):
                            d2 = (xs_val[i]-xs_val[j])**2 + (ys_val[i]-ys_val[j])**2
                            r_sum = rs_val[i] + rs_val[j]
                            if d2 < r_sum**2 - 1e-6:
                                valid = False
                    
                    if valid:
                        best_val = res.fun
                        best_x = res.x.copy()
                        
                        # Second pass with tighter tolerance from improved starting point
                        if time.time() < deadline - 15:
                            res2 = minimize(objective, best_x, method='SLSQP', 
                                         bounds=bounds,
                                         constraints=constraints,
                                         options={'maxiter': 100, 'ftol': 1e-12})
                            if res2.success and res2.fun < best_val:
                                # Quick validity check
                                w_val2 = res2.x[0]
                                h_val2 = 2.0 - w_val2
                                xs_val2 = res2.x[1:N+1]
                                ys_val2 = res2.x[N+1:2*N+1]
                                rs_val2 = res2.x[2*N+1:3*N+1]
                                
                                valid2 = True
                                for i in range(N):
                                    if xs_val2[i] < rs_val2[i] - 1e-6 or xs_val2[i] > w_val2 - rs_val2[i] + 1e-6:
                                        valid2 = False
                                    if ys_val2[i] < rs_val2[i] - 1e-6 or ys_val2[i] > h_val2 - rs_val2[i] + 1e-6:
                                        valid2 = False
                                for i in range(N):
                                    for j in range(i+1, N):
                                        d2 = (xs_val2[i]-xs_val2[j])**2 + (ys_val2[i]-ys_val2[j])**2
                                        r_sum = rs_val2[i] + rs_val2[j]
                                        if d2 < r_sum**2 - 1e-6:
                                            valid2 = False
                                            
                                if valid2:
                                    best_val = res2.fun
                                    best_x = res2.x.copy()
            except Exception as e:
                pass
    
    if best_x is None:
>>>>>>> REPLACE

