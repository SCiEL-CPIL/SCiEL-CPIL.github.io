import numpy as np
import scipy.integrate
import scipy.optimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    # --- 1. Define Validation Cosmologies ---
    # We need a dense grid to capture the "worst-case" relative error.
    # The task implies a range of CPL parameters. We use a broad but manageable grid.
    # w0 in [-1.5, 0.5], w1 in [-1.5, 1.5], z in [0.01, 10]
    
    # Redshifts: Log-spaced for low z, linear for high z to capture dynamic range
    z_vals = np.unique(np.concatenate([
        np.logspace(np.log10(0.01), np.log10(1.0), 10),
        np.linspace(1.0, 10.0, 15)
    ]))
    
    w0_vals = np.linspace(-1.5, 0.5, 12)
    w1_vals = np.linspace(-1.5, 1.5, 12)
    
    # Total combinations: 25 * 12 * 12 = 3600.
    # This is manageable for quad if we are careful.
    
    validation_cases = []
    for w0 in w0_vals:
        for w1 in w1_vals:
            for z in z_vals:
                validation_cases.append((w0, w1, z))
                
    n_cases = len(validation_cases)
    
    # --- 2. Precompute Exact Integrals ---
    # f(u) = 1 / E(z*u)
    # E(t)^2 = (1+t)^3 + (1+t)^(3(1+w0)) * exp(3*w1*(1 - 1/(1+t)))
    
    def calc_integrand(u, w0, w1, z):
        t = z * u
        # Avoid log(0) issues, though t>=0
        # Use log-sum-exp for stability
        log_t1 = 3.0 * np.log1p(t)
        log_t2 = 3.0 * (1.0 + w0) * np.log1p(t) + 3.0 * w1 * (1.0 - 1.0 / (1.0 + t))
        
        max_log = np.maximum(log_t1, log_t2)
        # If max_log is -inf (t=0 and w0 < -1?), handle carefully.
        # At t=0, log_t1 = 0, log_t2 = 0 (if w1 finite).
        # exp(log_t1 - max_log) + exp(log_t2 - max_log)
        
        sum_exp = np.exp(log_t1 - max_log) + np.exp(log_t2 - max_log)
        log_sum = max_log + np.log(sum_exp)
        
        return np.exp(-0.5 * log_sum)

    exact_ints = np.zeros(n_cases)
    w0_arr = np.zeros(n_cases)
    w1_arr = np.zeros(n_cases)
    z_arr = np.zeros(n_cases)
    
    for i, (w0, w1, z) in enumerate(validation_cases):
        w0_arr[i] = w0
        w1_arr[i] = w1
        z_arr[i] = z
        
        try:
            val, _ = scipy.integrate.quad(
                lambda u: calc_integrand(u, w0, w1, z), 
                0, 1, 
                limit=100, 
                epsabs=1e-10, 
                epsrel=1e-10
            )
            exact_ints[i] = val
        except Exception:
            # Fallback to simple trapz if quad fails (shouldn't happen often)
            us = np.linspace(0, 1, 1000)
            vals = np.array([calc_integrand(u, w0, w1, z) for u in us])
            exact_ints[i] = np.trapz(vals, us)
            
    # Check time
    if time.time() > deadline - 20:
        # If precomputation took too long, return GL
        gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
        u0 = 0.5 + 0.5 * gl_nodes
        w0_init = 0.5 * gl_weights
        return {"nodes": u0, "weights": w0_init}

    # --- 3. Define Objective Functions ---
    
    def compute_errors(params):
        """
        params: 16-dim vector.
        First 8: theta (nodes via sigmoid)
        Last 8: alpha (weights via softmax)
        """
        theta = params[:8]
        alpha = params[8:16]
        
        # Nodes in (0, 1)
        # Clip theta to avoid overflow in exp
        theta_clipped = np.clip(theta, -50, 50)
        u = 1.0 / (1.0 + np.exp(-theta_clipped))
        
        # Weights positive, sum to 1
        alpha_clipped = np.clip(alpha, -50, 50)
        exp_alpha = np.exp(alpha_clipped - np.max(alpha_clipped))
        w = exp_alpha / np.sum(exp_alpha)
        
        # Vectorized computation of f_vals
        # t shape: (n_cases, 8)
        t = z_arr[:, None] * u[None, :]
        
        log_t1 = 3.0 * np.log1p(t)
        log_t2 = 3.0 * (w0_arr)[:, None] * np.log1p(t) + 3.0 * w1_arr[:, None] * (1.0 - 1.0 / (1.0 + t))
        
        max_log = np.maximum(log_t1, log_t2)
        sum_exp = np.exp(log_t1 - max_log) + np.exp(log_t2 - max_log)
        log_sum = max_log + np.log(sum_exp)
        
        f_vals = np.exp(-0.5 * log_sum) # shape (n_cases, 8)
        
        # Quadrature sum: sum_i w_i f_i
        quad_sums = np.sum(f_vals * w[None, :], axis=1)
        
        # Relative errors
        # Avoid division by zero
        exact_ints_safe = np.maximum(exact_ints, 1e-30)
        rel_errors = np.abs(quad_sums - exact_ints_safe) / exact_ints_safe
        
        return rel_errors, u, w

    def objective_max(params):
        """Max relative error"""
        rel_errors, _, _ = compute_errors(params)
        return np.max(rel_errors)

    def objective_smooth(params):
        """
        Smooth surrogate: Sum of squared log-errors or SoftMax of errors.
        Using sum of squared relative errors is standard and differentiable.
        """
        rel_errors, _, _ = compute_errors(params)
        # Squared error is smooth
        return np.sum(rel_errors ** 2)

    # --- 4. Initialization ---
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    u0 = 0.5 + 0.5 * gl_nodes
    w0_init = 0.5 * gl_weights
    
    # Map to parameter space
    u0_safe = np.clip(u0, 1e-9, 1 - 1e-9)
    theta0 = np.log(u0_safe / (1 - u0_safe))
    
    w0_init_safe = np.clip(w0_init, 1e-9, 1.0)
    alpha0 = np.log(w0_init_safe)
    
    p0 = np.concatenate([theta0, alpha0])
    
    best_params = p0.copy()
    best_score = objective_max(p0)
    
    # --- 5. Optimization Phase 1: L-BFGS-B on Smooth Objective ---
    if time.time() < deadline - 15:
        try:
            # No bounds needed strictly due to sigmoid/softmax, but keep wide
            # to prevent extreme values that might cause numerical issues in exp
            res1 = scipy.optimize.minimize(
                objective_smooth,
                p0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res1.fun < 1e10: # Check for failure
                score1 = objective_max(res1.x)
                if score1 < best_score:
                    best_params = res1.x.copy()
                    best_score = score1
        except Exception:
            pass

    # --- 6. Optimization Phase 2: Powell on Max Error Objective ---
    if time.time() < deadline - 5:
        try:
            # Powell doesn't use gradients, good for non-smooth max function
            # We provide a box around the current best to help it
            # Or just start from best_params
            res2 = scipy.optimize.minimize(
                objective_max,
                best_params,
                method='Powell',
                options={'maxiter': 100, 'xtol': 1e-10, 'ftol': 1e-12}
            )
            
            if res2.fun < best_score:
                best_params = res2.x.copy()
                best_score = res2.fun
        except Exception:
            pass

    # --- 7. Final Extraction and Validation ---
    final_params = best_params
    theta_final = final_params[:8]
    alpha_final = final_params[8:16]
    
    u_final = 1.0 / (1.0 + np.exp(-np.clip(theta_final, -50, 50)))
    
    exp_alpha = np.exp(np.clip(alpha_final, -50, 50) - np.max(np.clip(alpha_final, -50, 50)))
    w_final = exp_alpha / np.sum(exp_alpha)
    
    # Ensure strictly increasing nodes
    # Sort if necessary, though sigmoid of optimized params should be monotonic
    # if the optimizer didn't cross them. But let's be safe.
    sort_idx = np.argsort(u_final)
    u_final = u_final[sort_idx]
    w_final = w_final[sort_idx]
    
    # Clip to [0,1] strictly
    u_final = np.clip(u_final, 1e-12, 1 - 1e-12)
    
    # Ensure weights sum to 1 exactly
    w_final = w_final / np.sum(w_final)
    
    return {"nodes": u_final, "weights": w_final}
