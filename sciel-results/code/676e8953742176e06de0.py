<<<<<<< SEARCH
    for start_idx in range(num_starts):
        if time.time() - start_time > time_limit:
            break
        
        # Staggered grid initialization
        cols = int(np.ceil(np.sqrt(n_circles * w / h)))
        rows = int(np.ceil(n_circles / cols))
        
        x0 = np.zeros(n_vars)
        for i in range(n_circles):
            col = i % cols
            row = i // cols
            
            # Staggered positions
            if row % 2 == 0:
                cx = (col + 0.5) * w / cols
            else:
                cx = (col + 1.0) * w / cols
                if cx > w:
                    cx = (col - 0.5) * w / cols
            
            cy = (row + 0.5) * h / rows
            
            # Add noise
            noise_x = 0.1 * w / cols
            noise_y = 0.1 * h / rows
            x0[2*i] = cx + np.random.uniform(-noise_x, noise_x)
            x0[2*i+1] = cy + np.random.uniform(-noise_y, noise_y)
            x0[2*i+2] = 0.3 * min(w/cols, h/rows)
            
        for i in range(n_circles):
            x0[2*i] = np.clip(x0[2*i], 0, w)
            x0[2*i+1] = np.clip(x0[2*i+1], 0, h)
            x0[2*i+2] = np.clip(x0[2*i+2], 0, max_r)
            
        penalty = 100.0
        
        def callback(Xk):
            return time.time() - start_time > time_limit
        
        try:
            res = minimize(
                lambda x: _objective_and_grad(x, w, h, n_circles, penalty)[0],
                x0,
                jac=lambda x: _objective_and_grad(x, w, h, n_circles, penalty)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-12, 'maxfun': 10000, 'callback': callback}
            )
            
            pos_r = res.x.copy()
            
            # Validate and fix
            if _validate_solution(pos_r, w, h, n_circles):
                val = -np.sum(pos_r[2::2])
                if val < best_val:
                    best_val = val
                    best_x = pos_r
            else:
                fixed_r = _fix_solution(pos_r, w, h, n_circles)
                if _validate_solution(fixed_r, w, h, n_circles):
                    val = -np.sum(fixed_r[2::2])
                    if val < best_val:
                        best_val = val
                        best_x = fixed_r
                        
        except Exception:
            pass
=======
    for start_idx in range(num_starts):
        if time.time() - start_time > time_limit:
            break
        
        x0 = _make_hex_initial(n_circles, w, h)
        
        # Add small noise to break symmetry
        x0[0::2] += np.random.uniform(-0.01*w, 0.01*w, n_circles)
        x0[1::2] += np.random.uniform(-0.01*h, 0.01*h, n_circles)
        
        for i in range(n_circles):
            x0[2*i] = np.clip(x0[2*i], 0, w)
            x0[2*i+1] = np.clip(x0[2*i+1], 0, h)
            x0[2*i+2] = np.clip(x0[2*i+2], 0, max_r)
            
        penalty = 100.0
        
        def callback(Xk):
            return time.time() - start_time > time_limit
        
        try:
            res = minimize(
                lambda x: _objective_and_grad(x, w, h, n_circles, penalty)[0],
                x0,
                jac=lambda x: _objective_and_grad(x, w, h, n_circles, penalty)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-12, 'maxfun': 10000, 'callback': callback}
            )
            
            pos_r = res.x.copy()
            
            # Second pass with higher penalty to tighten
            if time.time() - start_time < time_limit - 2:
                penalty2 = 1000.0
                try:
                    res2 = minimize(
                        lambda x: _objective_and_grad(x, w, h, n_circles, penalty2)[0],
                        pos_r,
                        jac=lambda x: _objective_and_grad(x, w, h, n_circles, penalty2)[1],
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 500, 'ftol': 1e-12, 'maxfun': 2000, 'callback': callback}
                    )
                    if np.isfinite(res2.fun):
                        pos_r = res2.x
                except Exception:
                    pass

            # Validate and fix
            if _validate_solution(pos_r, w, h, n_circles):
                val = -np.sum(pos_r[2::2])
                if val < best_val:
                    best_val = val
                    best_x = pos_r
            else:
                fixed_r = _fix_solution(pos_r, w, h, n_circles)
                if _validate_solution(fixed_r, w, h, n_circles):
                    val = -np.sum(fixed_r[2::2])
                    if val < best_val:
                        best_val = val
                        best_x = fixed_r
                        
        except Exception:
            pass
>>>>>>> REPLACE
