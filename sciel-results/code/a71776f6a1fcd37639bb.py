    target_ranks = [45, 44, 43, 42, 40, 38, 36]
    
    for idx, r_target in enumerate(target_ranks):
        if _time_left() < 20:
            break
        
        n_tr = 5 if idx == 0 else 3
        max_it = 150 if idx == 0 else 100
        result = _alt_minimize(U46, V46, W46, T, r_target, n_trials=n_tr, max_iter=max_it)
        if result is not None:
            U_t, V_t, W_t, err = result
            # Verify the result
            if _verify(U_t, V_t, W_t, n, m, p, tol=1e-6):
                # Check entries are half-integers
                U_check = np.round(U_t * 2) / 2
                V_check = np.round(V_t * 2) / 2
                W_check = np.round(W_t * 2) / 2
                if np.allclose(U_check, U_t, atol=1e-9) and \
                   np.allclose(V_check, V_t, atol=1e-9) and \
                   np.allclose(W_check, W_t, atol=1e-9):
                    return {"U": U_check.tolist(), "V": V_check.tolist(), "W": W_check.tolist()}
