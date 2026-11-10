<<<<<<< SEARCH
    # Run Product Structure Optimization
    for i, theta0 in enumerate(prod_inits):
        if _time_left() < 30:
            break
        try:
            res = opt.minimize(
                prod_objective,
                theta0,
                jac=prod_jac,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 20000}
            )
            
            # Extract best P from result
            theta_opt = res.x
            log_R = theta_opt[:n]
            log_C = theta_opt[n:]
            
            log_R -= np.max(log_R)
            R = np.exp(log_R)
            R /= np.sum(R)
            
            log_C -= np.max(log_C)
            C = np.exp(log_C)
            C /= np.sum(C)
            
            P_opt = np.outer(R, C)
            
            # Ensure validity
            if P_opt.shape != (n, n):
                continue
            if np.any(np.isnan(P_opt)) or np.any(np.isinf(P_opt)):
                continue
            if np.any(P_opt < -1e-9):
                continue
            if P_opt.sum() <= 1e-9:
                continue
                
            # Clip and normalize
            P_opt = np.clip(P_opt, 0, None)
            P_opt = P_opt / P_opt.sum()
            
            # Evaluate score
            ent = compute_entropies(P_opt.flatten())
            D = ent[0]
            M = ent[1]
            
            if M < 1e-9:
                continue
                
            score = D / M
            _update_best(P_opt, score)
            
        except Exception as e:
            pass
=======
    # Run Product Structure Optimization with multiple methods
    for i, theta0 in enumerate(prod_inits):
        if _time_left() < 30:
            break
        try:
            res = opt.minimize(
                prod_objective,
                theta0,
                jac=prod_jac,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 20000}
            )
            
            # Extract best P from result
            theta_opt = res.x
            log_R = theta_opt[:n]
            log_C = theta_opt[n:]
            
            log_R -= np.max(log_R)
            R = np.exp(log_R)
            R /= np.sum(R)
            
            log_C -= np.max(log_C)
            C = np.exp(log_C)
            C /= np.sum(C)
            
            P_opt = np.outer(R, C)
            
            # Ensure validity
            if P_opt.shape != (n, n):
                continue
            if np.any(np.isnan(P_opt)) or np.any(np.isinf(P_opt)):
                continue
            if np.any(P_opt < -1e-9):
                continue
            if P_opt.sum() <= 1e-9:
                continue
                
            # Clip and normalize
            P_opt = np.clip(P_opt, 0, None)
            P_opt = P_opt / P_opt.sum()
            
            # Evaluate score
            ent = compute_entropies(P_opt.flatten())
            D = ent[0]
            M = ent[1]
            
            if M < 1e-9:
                continue
                
            score = D / M
            _update_best(P_opt, score)
            
        except Exception as e:
            pass
        
        # Try SLSQP for refinement if time permits
        if _time_left() > 60:
            try:
                res2 = opt.minimize(
                    prod_objective,
                    theta0,
                    jac=prod_jac,
                    method='SLSQP',
                    options={'maxiter': 500, 'ftol': 1e-14, 'disp': False}
                )
                
                theta_opt2 = res2.x
                log_R2 = theta_opt2[:n]
                log_C2 = theta_opt2[n:]
                
                log_R2 -= np.max(log_R2)
                R2 = np.exp(log_R2)
                R2 /= np.sum(R2)
                
                log_C2 -= np.max(log_C2)
                C2 = np.exp(log_C2)
                C2 /= np.sum(C2)
                
                P_opt2 = np.outer(R2, C2)
                
                if P_opt2.shape != (n, n):
                    continue
                if np.any(np.isnan(P_opt2)) or np.any(np.isinf(P_opt2)):
                    continue
                if np.any(P_opt2 < -1e-9):
                    continue
                if P_opt2.sum() <= 1e-9:
                    continue
                    
                P_opt2 = np.clip(P_opt2, 0, None)
                P_opt2 = P_opt2 / P_opt2.sum()
                
                ent2 = compute_entropies(P_opt2.flatten())
                D2 = ent2[0]
                M2 = ent2[1]
                
                if M2 < 1e-9:
                    continue
                    
                score2 = D2 / M2
                _update_best(P_opt2, score2)
                
            except Exception as e:
                pass
>>>>>>> REPLACE
