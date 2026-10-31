<<<<<<< SEARCH
def _polish(pos, radii, deadline):
    """High-precision polishing with SLSQP hard constraints and gradient descent."""
    x = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
    for i in range(N):
        bounds.append((0.001, 0.5))
        
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # SLSQP with hard constraints - multiple attempts with different maxiter
    for attempt, maxiter in enumerate([300, 500, 800]):
        if time.time() > deadline - 15:
            break
        try:
            cons = _slsqp_constraints(pos, radii)
            result = minimize(lambda x: -np.sum(np.maximum(x[2*N:], 1e-6)), x, method='SLSQP',
                              bounds=bounds, constraints=cons,
                              options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10})
            if result.success or result.fun < -np.sum(np.maximum(x[2*N:], 1e-6)) + 1e-10:
                x_new = result.x
                pos_slq = x_new[:2*N].reshape(N, 2)
                rad_slq = np.maximum(x_new[2*N:], 1e-4)
                pos_pp, radii_pp = _post_process(pos_slq, rad_slq)
                if _is_valid(pos_pp, radii_pp):
                    score = np.sum(radii_pp)
                    if score > best_score:
                        best_score = score
                        best_pos = pos_pp.copy()
                        best_radii = radii_pp.copy()
                        x = x_new.copy()
        except Exception:
            pass
    
    # Gradient descent for fine-tuning
    if time.time() < deadline - 8:
        try:
            pos_gd, rad_gd = _gradient_descent(best_pos, best_radii, n_steps=4000, lr_init=0.004, lr_decay=0.9995, seed=42, deadline=deadline - 6)
            pos_pp, radii_pp = _post_process(pos_gd, rad_gd)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass
    
    # L-BFGS-B with high penalty for final fine-tuning
    if time.time() < deadline - 5:
        try:
            x_final = np.concatenate([best_pos.flatten(), best_radii])
            result = minimize(_objective, x_final, method='L-BFGS-B', bounds=bounds,
                              args=(5000.0,),
                              options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-10})
            if result.fun < _objective(x_final, 5000.0) - 1e-12:
                x = result.x
                pos_lbfgs = x[:2*N].reshape(N, 2)
                rad_lbfgs = np.maximum(x[2*N:], 1e-4)
                pos_pp, radii_pp = _post_process(pos_lbfgs, rad_lbfgs)
                if _is_valid(pos_pp, radii_pp):
                    score = np.sum(radii_pp)
                    if score > best_score:
                        best_score = score
                        best_pos = pos_pp.copy()
                        best_radii = radii_pp.copy()
        except Exception:
            pass
    
    # Second SLSQP pass from improved position
    if time.time() < deadline - 5:
        try:
            x2 = np.concatenate([best_pos.flatten(), best_radii])
            cons2 = _slsqp_constraints(best_pos, best_radii)
            result2 = minimize(lambda x: -np.sum(np.maximum(x[2*N:], 1e-6)), x2, method='SLSQP',
                               bounds=bounds, constraints=cons2,
                               options={'maxiter': 600, 'ftol': 1e-14, 'gtol': 1e-12})
            if result2.success or result2.fun < -np.sum(np.maximum(x2[2*N:], 1e-6)) + 1e-10:
                x = result2.x
                pos_s2 = x[:2*N].reshape(N, 2)
                rad_s2 = np.maximum(x[2*N:], 1e-4)
                pos_pp, radii_pp = _post_process(pos_s2, rad_s2)
                if _is_valid(pos_pp, radii_pp):
                    score = np.sum(radii_pp)
                    if score > best_score:
                        best_score = score
                        best_pos = pos_pp.copy()
                        best_radii = radii_pp.copy()
        except Exception:
            pass
    
    return best_pos, best_radii
=======
def _polish(pos, radii, deadline):
    """High-precision polishing with SLSQP hard constraints, Gradient Descent, and LP radius optimization."""
    x = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
    for i in range(N):
        bounds.append((0.001, 0.5))
        
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # Helper to update best if LP gives a better result
    def try_update(pos_cand, rad_cand):
        nonlocal best_pos, best_radii, best_score
        # Try LP optimization on this position
        try:
            rad_lp = _lp_radius_optimize(pos_cand)
            # Ensure LP radii are valid
            pos_pp, radii_pp = _post_process(pos_cand, rad_lp)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
                    return
        except Exception:
            pass
        
        # Fallback to post-process original radii
        pos_pp, radii_pp = _post_process(pos_cand, rad_cand)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()

    # SLSQP with hard constraints - multiple attempts with different maxiter
    for attempt, maxiter in enumerate([300, 500, 800]):
        if time.time() > deadline - 15:
            break
        try:
            cons = _slsqp_constraints(best_pos, best_radii)
            x_current = np.concatenate([best_pos.flatten(), best_radii])
            result = minimize(lambda x: -np.sum(np.maximum(x[2*N:], 1e-6)), x_current, method='SLSQP',
                              bounds=bounds, constraints=cons,
                              options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10})
            if result.success or result.fun < -np.sum(np.maximum(x_current[2*N:], 1e-6)) + 1e-10:
                x_new = result.x
                pos_slq = x_new[:2*N].reshape(N, 2)
                rad_slq = np.maximum(x_new[2*N:], 1e-4)
                try_update(pos_slq, rad_slq)
        except Exception:
            pass
    
    # Gradient descent for fine-tuning
    if time.time() < deadline - 8:
        try:
            pos_gd, rad_gd = _gradient_descent(best_pos, best_radii, n_steps=4000, lr_init=0.004, lr_decay=0.9995, seed=42, deadline=deadline - 6)
            try_update(pos_gd, rad_gd)
        except Exception:
            pass
    
    # L-BFGS-B with high penalty for final fine-tuning
    if time.time() < deadline - 5:
        try:
            x_final = np.concatenate([best_pos.flatten(), best_radii])
            result = minimize(_objective, x_final, method='L-BFGS-B', bounds=bounds,
                              args=(5000.0,),
                              options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-10})
            if result.fun < _objective(x_final, 5000.0) - 1e-12:
                x = result.x
                pos_lbfgs = x[:2*N].reshape(N, 2)
                rad_lbfgs = np.maximum(x[2*N:], 1e-4)
                try_update(pos_lbfgs, rad_lbfgs)
        except Exception:
            pass
    
    # Second SLSQP pass from improved position
    if time.time() < deadline - 5:
        try:
            x2 = np.concatenate([best_pos.flatten(), best_radii])
            cons2 = _slsqp_constraints(best_pos, best_radii)
            result2 = minimize(lambda x: -np.sum(np.maximum(x[2*N:], 1e-6)), x2, method='SLSQP',
                               bounds=bounds, constraints=cons2,
                               options={'maxiter': 600, 'ftol': 1e-14, 'gtol': 1e-12})
            if result2.success or result2.fun < -np.sum(np.maximum(x2[2*N:], 1e-6)) + 1e-10:
                x = result2.x
                pos_s2 = x[:2*N].reshape(N, 2)
                rad_s2 = np.maximum(x[2*N:], 1e-4)
                try_update(pos_s2, rad_s2)
        except Exception:
            pass
            
    # Final LP pass on the best found position
    if time.time() < deadline - 2:
        try:
            rad_lp = _lp_radius_optimize(best_pos)
            pos_pp, radii_pp = _post_process(best_pos, rad_lp)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass
    
    return best_pos, best_radii
>>>>>>> REPLACE

