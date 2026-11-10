<<<<<<< SEARCH
    # Use Nelder-Mead with a few restarts
    np.random.seed(42)
    
    for restart in range(5):
        if time.time() > deadline - 60:
            break
        
        if restart == 0:
            x_start = x0.copy()
        else:
            x_start = x0 + np.random.randn(24) * 0.3
        
        try:
            result = minimize(
                score_func, x_start,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-9}
            )
            
            if result.fun < best_s:
                best_s = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
=======
    # Use Nelder-Mead with limited iterations to ensure speed
    np.random.seed(42)
    
    for restart in range(3):
        if time.time() > deadline - 30:
            break
        
        if restart == 0:
            x_start = x0.copy()
        else:
            x_start = x0 + np.random.randn(24) * 0.2
        
        try:
            result = minimize(
                score_func, x_start,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            if result.fun < best_s:
                best_s = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
>>>>>>> REPLACE
