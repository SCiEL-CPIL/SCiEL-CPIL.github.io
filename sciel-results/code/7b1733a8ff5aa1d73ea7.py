<<<<<<< SEARCH
    # 6. Final Extraction
    theta_final = best_params[:8]
    alpha_final = best_params[8:16]
    
    u_final = 1.0 / (1.0 + np.exp(-np.clip(theta_final, -50, 50)))
    exp_alpha = np.exp(np.clip(alpha_final, -50, 50) - np.max(np.clip(alpha_final, -50, 50)))
    w_final = exp_alpha / np.sum(exp_alpha)
    
    # Enforce strictly increasing nodes
    sort_idx = np.argsort(u_final)
    u_final = u_final[sort_idx]
    w_final = w_final[sort_idx]
    
    # Clip to [0,1]
    u_final = np.clip(u_final, 1e-12, 1 - 1e-12)
    
    # Ensure weights sum to 1
    w_final = w_final / np.sum(w_final)
    
    # Compare with GL
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    u_gl = 0.5 + 0.5 * gl_nodes
    w_gl = 0.5 * gl_weights
    
    # Compute error for GL
    t_gl = z_arr[:, None] * u_gl[None, :]
    log_term1 = 3.0 * np.log1p(t_gl)
    log_term2 = (3.0 + 3.0 * w0_arr)[:, None] * np.log1p(t_gl) + 3.0 * w1_arr[:, None] * (t_gl / (1.0 + t_gl))
    m = np.maximum(log_term1, log_term2)
    sum_exp = np.exp(log_term1 - m) + np.exp(log_term2 - m)
    log_E_sq = m + np.log(sum_exp)
    f_vals_gl = np.exp(-0.5 * log_E_sq)
    quad_sums_gl = np.sum(f_vals_gl * w_gl[None, :], axis=1)
    exact_ints_safe = np.maximum(exact_ints, 1e-30)
    gl_err = np.max(np.abs(quad_sums_gl - exact_ints_safe) / exact_ints_safe)
    
    # Compute error for optimized
    t_opt = z_arr[:, None] * u_final[None, :]
    log_term1 = 3.0 * np.log1p(t_opt)
    log_term2 = (3.0 + 3.0 * w0_arr)[:, None] * np.log1p(t_opt) + 3.0 * w1_arr[:, None] * (t_opt / (1.0 + t_opt))
    m = np.maximum(log_term1, log_term2)
    sum_exp = np.exp(log_term1 - m) + np.exp(log_term2 - m)
    log_E_sq = m + np.log(sum_exp)
    f_vals_opt = np.exp(-0.5 * log_E_sq)
    quad_sums_opt = np.sum(f_vals_opt * w_final[None, :], axis=1)
    opt_err = np.max(np.abs(quad_sums_opt - exact_ints_safe) / exact_ints_safe)
    
    if opt_err > gl_err:
        u_final = u_gl
        w_final = w_gl

    return {"nodes": u_final, "weights": w_final}
=======
    # 6. Final Extraction
    theta_final = best_params[:8]
    alpha_final = best_params[8:16]
    
    u_final = 1.0 / (1.0 + np.exp(-np.clip(theta_final, -50, 50)))
    exp_alpha = np.exp(np.clip(alpha_final, -50, 50) - np.max(np.clip(alpha_final, -50, 50)))
    w_final = exp_alpha / np.sum(exp_alpha)
    
    # Enforce strictly increasing nodes
    sort_idx = np.argsort(u_final)
    u_final = u_final[sort_idx]
    w_final = w_final[sort_idx]
    
    # Clip to [0,1]
    u_final = np.clip(u_final, 1e-12, 1 - 1e-12)
    
    # Ensure weights sum to 1
    w_final = w_final / np.sum(w_final)
    
    # Compare with GL
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    u_gl = 0.5 + 0.5 * gl_nodes
    w_gl = 0.5 * gl_weights
    
    # Compute error for GL
    t_gl = z_arr[:, None] * u_gl[None, :]
    log_term1 = 3.0 * np.log1p(t_gl)
    log_term2 = (3.0 + 3.0 * w0_arr)[:, None] * np.log1p(t_gl) + 3.0 * w1_arr[:, None] * (t_gl / (1.0 + t_gl))
    m = np.maximum(log_term1, log_term2)
    sum_exp = np.exp(log_term1 - m) + np.exp(log_term2 - m)
    log_E_sq = m + np.log(sum_exp)
    f_vals_gl = np.exp(-0.5 * log_E_sq)
    quad_sums_gl = np.sum(f_vals_gl * w_gl[None, :], axis=1)
    exact_ints_safe = np.maximum(exact_ints, 1e-30)
    gl_err = np.max(np.abs(quad_sums_gl - exact_ints_safe) / exact_ints_safe)
    
    # Compute error for optimized
    t_opt = z_arr[:, None] * u_final[None, :]
    log_term1 = 3.0 * np.log1p(t_opt)
    log_term2 = (3.0 + 3.0 * w0_arr)[:, None] * np.log1p(t_opt) + 3.0 * w1_arr[:, None] * (t_opt / (1.0 + t_opt))
    m = np.maximum(log_term1, log_term2)
    sum_exp = np.exp(log_term1 - m) + np.exp(log_term2 - m)
    log_E_sq = m + np.log(sum_exp)
    f_vals_opt = np.exp(-0.5 * log_E_sq)
    quad_sums_opt = np.sum(f_vals_opt * w_final[None, :], axis=1)
    opt_err = np.max(np.abs(quad_sums_opt - exact_ints_safe) / exact_ints_safe)
    
    if opt_err > gl_err:
        u_final = u_gl
        w_final = w_gl

    return {"nodes": u_final, "weights": w_final}
>>>>>>> REPLACE
