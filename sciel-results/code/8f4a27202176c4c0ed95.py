    if time.time() - start_time < time_limit:
        try:
            from scipy.optimize import basin_hopping
            
            def bh_obj(x):
                b = x.reshape(N, 2)
                b = clamp_bary(b)
                return -np.min(calc_areas_bary(b))
            
            bounds = [(0, 1)] * (N * 2)
            
            # Use Nelder-Mead as the local minimizer
            result_bh = basin_hopping(
                bh_obj,
                bary_current.flatten(),
                minimizer_kwargs={'method': 'Nelder-Mead', 'options': {'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}},
                stepsize=0.1,
                niter=50, # Limit iterations to fit time
                seed=42
            )
            
            new_bary = clamp_bary(result_bh.x.reshape(N, 2))
            new_val = calc_min_area(new_bary)
            if new_val > curr_val:
                bary_current = new_bary
                curr_val = new_val
        except Exception:
            pass
    