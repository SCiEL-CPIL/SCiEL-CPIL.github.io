<<<<<<< SEARCH
        # Step 1: L-BFGS-B with soft objective (annealing beta)
        # Start with moderate beta to explore, increase to refine
        betas = [30.0, 80.0, 150.0, 300.0]
        
        for beta in betas:
            if time.time() > start_time + time_limit - 4:
                break
            try:
                res = opt.minimize(
                    lambda x, b=beta: _soft_objective(x, b),
                    current.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 80, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 2000}
                )
                current = res.x.reshape(-1, 2)
            except Exception:
                pass
        
        # Step 2: Powell with hard objective
        if time.time() > start_time + time_limit - 2:
            break
        try:
            res = opt.minimize(
                _hard_objective,
                current.flatten(),
                method='Powell',
                options={'maxiter': 150, 'xtol': 1e-14, 'ftol': 1e-14}
            )
            candidate = res.x.reshape(-1, 2)
            score = compute_score(candidate)
            
            if score > best_score:
                best_score = score
                best_points = candidate.copy()
        except Exception:
            pass
        
        # Step 3: Coordinate descent for further refinement
        if time.time() > start_time + time_limit - 1:
            break
        try:
            candidate = _coordinate_descent(current, max_iter=150, initial_step=0.02)
            score = compute_score(candidate)
            
            if score > best_score:
                best_score = score
                best_points = candidate.copy()
        except Exception:
            pass
=======
        # Step 1: L-BFGS-B with soft objective (annealing beta)
        # Start with moderate beta to explore, increase to refine
        betas = [30.0, 80.0, 150.0, 300.0, 500.0]
        
        for beta in betas:
            if time.time() > start_time + time_limit - 4:
                break
            try:
                res = opt.minimize(
                    lambda x, b=beta: _soft_objective(x, b),
                    current.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 120, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 3000}
                )
                current = res.x.reshape(-1, 2)
            except Exception:
                pass
        
        # Step 2: Nelder-Mead with hard objective
        if time.time() > start_time + time_limit - 3:
            break
        try:
            res = opt.minimize(
                _hard_objective,
                current.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
            )
            candidate = res.x.reshape(-1, 2)
            score = compute_score(candidate)
            
            if score > best_score:
                best_score = score
                best_points = candidate.copy()
                current = candidate.copy()
        except Exception:
            pass
        
        # Step 3: Powell with hard objective
        if time.time() > start_time + time_limit - 2:
            break
        try:
            res = opt.minimize(
                _hard_objective,
                current.flatten(),
                method='Powell',
                options={'maxiter': 200, 'xtol': 1e-14, 'ftol': 1e-14}
            )
            candidate = res.x.reshape(-1, 2)
            score = compute_score(candidate)
            
            if score > best_score:
                best_score = score
                best_points = candidate.copy()
        except Exception:
            pass
        
        # Step 4: Coordinate descent for further refinement
        if time.time() > start_time + time_limit - 1:
            break
        try:
            candidate = _coordinate_descent(current, max_iter=200, initial_step=0.02)
            score = compute_score(candidate)
            
            if score > best_score:
                best_score = score
                best_points = candidate.copy()
        except Exception:
            pass
>>>>>>> REPLACE

