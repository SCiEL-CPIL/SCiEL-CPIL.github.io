<<<<<<< SEARCH
    # Final refinement with best parameters
    if time.time() < deadline - 10:
        try:
            result_final = minimize(
                _objective,
                best_params,
                args=(i_flat, j_flat, N, 56),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if result_final.fun < best_vol:
                best_params = result_final.x.copy()
                best_vol = result_final.fun
        except Exception:
            pass
=======
    # Final refinement with best parameters
    if time.time() < deadline - 10:
        try:
            result_final = minimize(
                _objective,
                best_params,
                args=(i_flat, j_flat, N, 64),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if result_final.fun < best_vol:
                best_params = result_final.x.copy()
                best_vol = result_final.fun
        except Exception:
            pass
    
    # Local perturbation search around best solution
    if time.time() < deadline - 20:
        perturbation_scales = [0.01, 0.005, 0.001]
        for scale in perturbation_scales:
            if time.time() > deadline - 10:
                break
            for _ in range(5):
                if time.time() > deadline - 10:
                    break
                perturbed = best_params + scale * rng.standard_normal(128)
                try:
                    p_result = minimize(
                        _objective,
                        perturbed,
                        args=(i_flat, j_flat, N, 56),
                        method='L-BFGS-B',
                        options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
                    )
                    if p_result.fun < best_vol:
                        best_params = p_result.x.copy()
                        best_vol = p_result.fun
                except Exception:
                    continue
>>>>>>> REPLACE
