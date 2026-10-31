<<<<<<< SEARCH
    # 4. Final high-precision polish
    if best_bary is not None and time.time() < deadline - 5:
        try:
            bounds = [(0, 1)] * (N * 2)
            result_slsqp = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-15}
            )
            update_best(clamp_bary(result_slsqp.x.reshape(N, 2)))
        except Exception:
            pass
        
        if best_bary is not None and time.time() < deadline - 3:
            # Epigraph-based final polish
            try:
                current_min = -hard_min_obj(best_bary.flatten())
                x_init = np.concatenate([best_bary.flatten(), [current_min * 0.95]])
                epigraph_bounds = bounds + [(0.0, 0.1)]
                
                result_ep_final = minimize(
                    epigraph_obj,
                    x_init,
                    method='L-BFGS-B',
                    bounds=epigraph_bounds,
                    options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                
                if result_ep_final.success:
                    new_bary = clamp_bary(result_ep_final.x[:N*2].reshape(N, 2))
                    update_best(new_bary)
            except Exception:
                pass
        
        if best_bary is not None and time.time() < deadline - 1.5:
            try:
                result_nm = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-18, 'adaptive': True}
                )
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass
        
        # One more SLSQP pass
        if best_bary is not None and time.time() < deadline - 0.5:
            try:
                bounds = [(0, 1)] * (N * 2)
                result_final = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 2000, 'ftol': 1e-16}
                )
                update_best(clamp_bary(result_final.x.reshape(N, 2)))
            except Exception:
                pass
=======
    # 4. Final high-precision polish with active-set focus
    if best_bary is not None and time.time() < deadline - 5:
        # Active-set final refinement
        best_bary = active_set_refine(best_bary, time_limit=3.0)
        update_best(best_bary)
        
        if best_bary is not None and time.time() < deadline - 3:
            try:
                bounds = [(0, 1)] * (N * 2)
                result_slsqp = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 800, 'ftol': 1e-15}
                )
                update_best(clamp_bary(result_slsqp.x.reshape(N, 2)))
            except Exception:
                pass
        
        if best_bary is not None and time.time() < deadline - 1.5:
            try:
                result_nm = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-18, 'adaptive': True}
                )
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass
>>>>>>> REPLACE
