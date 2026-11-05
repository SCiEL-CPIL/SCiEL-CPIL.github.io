<<<<<<< SEARCH
    # Optimize each candidate
    for mats, ths in candidates:
        if time.time() > deadline - 10:
            break
        
        x_opt = _optimize_stack(mats, ths, problem, wls, n_mat, deadline - 5, max_layer, max_total)
        
        # Update best
        update_best(mats, x_opt)
        
        # Also try with the original thicknesses in case optimization failed
        update_best(mats, ths)
=======
    # Optimize each candidate
    for mats, ths in candidates:
        if time.time() > deadline - 10:
            break
        
        x_opt = _optimize_stack(mats, ths, problem, wls, n_mat, deadline - 5, max_layer, max_total)
        
        # Update best
        update_best(mats, x_opt)
        
        # Also try with the original thicknesses in case optimization failed
        update_best(mats, ths)
    
    # Final polish pass on the best solution with tighter tolerance
    if best_mats is not None and time.time() < deadline - 2:
        try:
            lo = np.zeros(len(best_ths))
            hi = np.full(len(best_ths), max_layer)
            
            def objective_polish(th):
                th = _project_to_feasible(th, max_layer, max_total)
                return _mean_reflectance_percent(best_mats, list(th), problem, wls, n_mat)
            
            res = minimize(objective_polish, np.array(best_ths), method='L-BFGS-B', 
                          bounds=list(zip(lo, hi)),
                          options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-10})
            if res.success:
                best_ths = list(_project_to_feasible(res.x, max_layer, max_total))
                val = update_best(best_mats, best_ths)
        except Exception:
            pass
>>>>>>> REPLACE
