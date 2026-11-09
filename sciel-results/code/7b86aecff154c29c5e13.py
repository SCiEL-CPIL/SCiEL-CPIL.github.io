<<<<<<< SEARCH
    # Phase 5: Final fine polish
    if time.time() < deadline - 5.0:
        try:
            res_final = minimize(
                _objective_exact,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 30000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            score_f = -res_final.fun
            if score_f > best_score:
                best_params = res_final.x
                best_score = score_f
        except Exception:
            pass
=======
    # Phase 5: Final fine polish with multiple passes
    if time.time() < deadline - 5.0:
        try:
            # First polish pass
            res_final = minimize(
                _objective_exact,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 30000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            score_f = -res_final.fun
            if score_f > best_score:
                best_params = res_final.x
                best_score = score_f
        except Exception:
            pass

    # Phase 6: Ultra-fine polish if time remains
    if time.time() < deadline - 3.0:
        try:
            res_final2 = minimize(
                _objective_exact,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 50000, 'xatol': 1e-15, 'fatol': 1e-17, 'adaptive': True}
            )
            score_f2 = -res_final2.fun
            if score_f2 > best_score:
                best_params = res_final2.x
                best_score = score_f2
        except Exception:
            pass
>>>>>>> REPLACE
