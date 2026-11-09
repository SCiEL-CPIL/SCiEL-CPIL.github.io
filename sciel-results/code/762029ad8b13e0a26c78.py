<<<<<<< SEARCH
    # Additional high-precision polish with Nelder-Mead on the best candidate
    if time.time() < deadline - 30:
        try:
            def nm_objective(params):
                P = np.exp(params).reshape(n, n)
                P = np.maximum(P, 0)
                total = P.sum()
                if total <= 1e-9:
                    return 1e10
                P = P / total
                return -compute_ratio(P)
            
            P_init_nm = np.maximum(P_best, 1e-12)
            params_init_nm = np.log(P_init_nm.flatten())
            
            res_nm = minimize(
                nm_objective,
                params_init_nm,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            P_nm = np.exp(res_nm.x).reshape(n, n)
            P_nm = np.maximum(P_nm, 0)
            total_nm = P_nm.sum()
            if total_nm > 1e-9:
                P_nm /= total_nm
                ratio_nm = compute_ratio(P_nm)
                if ratio_nm > best_ratio:
                    best_ratio = ratio_nm
                    P_best = P_nm.copy()
        except Exception:
            pass
=======
    # Additional high-precision polish with Nelder-Mead on the best candidate
    if time.time() < deadline - 30:
        try:
            def nm_objective(params):
                P = np.exp(params).reshape(n, n)
                P = np.maximum(P, 0)
                total = P.sum()
                if total <= 1e-9:
                    return 1e10
                P = P / total
                return -compute_ratio(P)
            
            P_init_nm = np.maximum(P_best, 1e-12)
            params_init_nm = np.log(P_init_nm.flatten())
            
            res_nm = minimize(
                nm_objective,
                params_init_nm,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True, 'maxfev': 5000}
            )
            
            P_nm = np.exp(res_nm.x).reshape(n, n)
            P_nm = np.maximum(P_nm, 0)
            total_nm = P_nm.sum()
            if total_nm > 1e-9:
                P_nm /= total_nm
                ratio_nm = compute_ratio(P_nm)
                if ratio_nm > best_ratio:
                    best_ratio = ratio_nm
                    P_best = P_nm.copy()
        except Exception:
            pass
    
    # Final SLSQP refinement on the very best candidate found
    if time.time() < deadline - 15:
        try:
            P_final_init = np.maximum(P_best, 1e-12)
            params_final = np.log(P_final_init.flatten())
            
            def final_slsqp_obj(params):
                P = np.exp(params).reshape(n, n)
                P = np.maximum(P, 0)
                total_p = P.sum()
                if total_p <= 1e-9:
                    return 1e10
                P = P / total_p
                return -compute_ratio(P)
            
            res_final = minimize(
                final_slsqp_obj,
                params_final,
                method='SLSQP',
                bounds=[(-50, 50)] * (n*n),
                options={'maxiter': 500, 'ftol': 1e-14, 'disp': False}
            )
            
            P_final = np.exp(res_final.x).reshape(n, n)
            P_final = np.maximum(P_final, 0)
            total_final = P_final.sum()
            if total_final > 1e-9:
                P_final /= total_final
                ratio_final = compute_ratio(P_final)
                if ratio_final > best_ratio:
                    best_ratio = ratio_final
                    P_best = P_final.copy()
        except Exception:
            pass
>>>>>>> REPLACE

