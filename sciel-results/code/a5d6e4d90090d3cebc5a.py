<<<<<<< SEARCH
    def objective(coeffs):
        """Max relative error over all test models and a values."""
        max_err = 0.0
        for (w0, wa, Om0) in test_models:
            if (w0, wa, Om0) not in ref_data:
                continue
            D_ref, f_ref = ref_data[(w0, wa, Om0)]
            
            # Compute approximations vectorized if possible, but loops are fine for 50 points
            # To speed up, we can vectorize the gamma calculation
            a_arr = a_values
            Om_arr = np.array([_Om_of_a(ai, Om0, w0, wa) for ai in a_arr])
            q_arr = 1.0 - Om_arr
            u_arr = 1.0 + w0 + wa * (1.0 - a_arr)
            
            c0, c1, c2, c3, c4, c5 = coeffs
            gamma_arr = c0 + c1*q_arr + c2*q_arr**2 + c3*u_arr + c4*wa*a_arr + c5*u_arr*q_arr
            gamma_arr = np.clip(gamma_arr, 0.1, 1.2)
            
            f_approx_arr = Om_arr ** gamma_arr
            
            # Calculate D_approx
            # D(a) = exp(-int_a^1 f(a')/a' da') ... wait, the task says exp(-integral_ln(a)^0 f(exp(x)) dx)
            # Let x = ln(a'), dx = da'/a'.
            # Integral from ln(a) to 0 of f(exp(x)) dx
            # This is equivalent to integral from a to 1 of f(a')/a' da'
            
            # We can compute this integral numerically for each a in the array
            # Since a_values is sorted, we can use cumulative trapezoidal integration from the end
            # f_over_a = f_approx_arr / a_arr
            # integral_i = sum_{j>i} 0.5 * (f_j/a_j + f_i/a_i) * (a_j - a_i) ... 
            # Actually, let's just use quad for each point to be safe, or a simple trapezoidal rule on the grid.
            # Trapezoidal rule is fast and accurate enough for optimization.
            
            f_over_a = f_approx_arr / a_arr
            # We want integral from a_i to 1.
            # Let's compute cumulative integral from 1 down to a_min.
            # da = a[1:] - a[:-1]
            # avg_f = 0.5 * (f_over_a[1:] + f_over_a[:-1])
            # step_int = avg_f * da
            # Cumulative sum from the end.
            
            da = np.diff(a_arr)
            avg_f = 0.5 * (f_over_a[1:] + f_over_a[:-1])
            step_int = avg_f * da
            
            # cumulative integral from a_i to a_{i+1} is step_int[i]
            # Integral from a_i to 1 is sum(step_int[i:] )
            # We can use cumsum from the end
            cum_int = np.zeros_like(a_arr)
            cum_int[-2] = step_int[-1]
            for i in range(len(step_int)-2, -1, -1):
                cum_int[i] = cum_int[i+1] + step_int[i]
                
            D_approx_arr = np.exp(-cum_int)
            
            # Relative errors
            err_f = np.abs(f_approx_arr - f_ref) / np.abs(f_ref)
            err_D = np.abs(D_approx_arr - D_ref) / np.abs(D_ref)
            
            # Handle potential zeros in reference (unlikely for f, D is normalized to 1 at a=1)
            err_f = np.nan_to_num(err_f, nan=0.0, posinf=1.0)
            err_D = np.nan_to_num(err_D, nan=0.0, posinf=1.0)
            
            max_err = max(max_err, np.max(err_f), np.max(err_D))
        
        return max_err
=======
    def objective(coeffs):
        """Max relative error over all test models and a values, with penalty for gamma out of bounds."""
        max_err = 0.0
        gamma_violation = 0.0
        for (w0, wa, Om0) in test_models:
            if (w0, wa, Om0) not in ref_data:
                continue
            D_ref, f_ref = ref_data[(w0, wa, Om0)]
            
            # Compute approximations vectorized if possible, but loops are fine for 50 points
            # To speed up, we can vectorize the gamma calculation
            a_arr = a_values
            Om_arr = np.array([_Om_of_a(ai, Om0, w0, wa) for ai in a_arr])
            q_arr = 1.0 - Om_arr
            u_arr = 1.0 + w0 + wa * (1.0 - a_arr)
            
            c0, c1, c2, c3, c4, c5 = coeffs
            gamma_arr = c0 + c1*q_arr + c2*q_arr**2 + c3*u_arr + c4*wa*a_arr + c5*u_arr*q_arr
            
            # Check for raw gamma violations
            # The evaluator requires gamma in [0.1, 1.2].
            # We penalize any deviation from this range heavily.
            lower_violation = np.maximum(0.1 - gamma_arr, 0.0)
            upper_violation = np.maximum(gamma_arr - 1.2, 0.0)
            max_violation = np.max(np.concatenate([lower_violation, upper_violation]))
            gamma_violation = max(gamma_violation, max_violation)
            
            # For the error calculation, we must use the raw gamma if it's in bounds.
            # If it's out of bounds, the evaluator will fail, so we return a high error.
            # However, to guide the optimizer, we can compute the error using the clipped gamma
            # but add the violation penalty.
            gamma_clipped = np.clip(gamma_arr, 0.1, 1.2)
            f_approx_arr = Om_arr ** gamma_clipped
            
            # Calculate D_approx
            f_over_a = f_approx_arr / a_arr
            da = np.diff(a_arr)
            avg_f = 0.5 * (f_over_a[1:] + f_over_a[:-1])
            step_int = avg_f * da
            
            cum_int = np.zeros_like(a_arr)
            if len(step_int) > 0:
                cum_int[-2] = step_int[-1]
                for i in range(len(step_int)-2, -1, -1):
                    cum_int[i] = cum_int[i+1] + step_int[i]
                
            D_approx_arr = np.exp(-cum_int)
            
            # Relative errors
            err_f = np.abs(f_approx_arr - f_ref) / np.abs(f_ref)
            err_D = np.abs(D_approx_arr - D_ref) / np.abs(D_ref)
            
            # Handle potential zeros in reference
            err_f = np.nan_to_num(err_f, nan=0.0, posinf=1.0)
            err_D = np.nan_to_num(err_D, nan=0.0, posinf=1.0)
            
            max_err = max(max_err, np.max(err_f), np.max(err_D))
        
        # If gamma is out of bounds, the score is effectively 0 (or very bad).
        # We return a large value to push the optimizer away.
        if gamma_violation > 1e-6:
            return 100.0 + gamma_violation * 1000.0
            
        return max_err
>>>>>>> REPLACE

We also need to ensure that the final returned coefficients are safe. If the optimizer finds a solution with a tiny violation that was penalized but still slightly out, or if the penalty wasn't strong enough, we should clamp the coefficients or revert to seed if the raw check fails.

