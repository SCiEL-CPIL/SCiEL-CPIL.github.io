import numpy as np
import scipy.integrate
import scipy.optimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 82.0

    # 1. Define Validation Cosmologies
    # We want to capture the worst-case relative error across a range of CPL parameters
    # and redshifts.
    # z in [0.01, 10]
    # w0 in [-1.5, 0.5]
    # w1 in [-1.5, 1.5]
    
    z_vals = np.unique(np.concatenate([
        np.logspace(np.log10(0.01), np.log10(1.0), 15),
        np.linspace(1.0, 10.0, 20)
    ]))
    
    w0_vals = np.linspace(-1.5, 0.5, 10)
    w1_vals = np.linspace(-1.5, 1.5, 10)
    
    validation_cases = []
    for w0 in w0_vals:
        for w1 in w1_vals:
            for z in z_vals:
                validation_cases.append((w0, w1, z))
                
    n_cases = len(validation_cases)
    
    # 2. Precompute Exact Integrals
    # dL/(c/H0) = (1+z) * Integral_0^z dz' / E(z')
    # Change variable u = z'/z => dz' = z du
    # dL/(c/H0) = (1+z) * z * Integral_0^1 du / E(z*u)
    # The task formula is: (1+z)*z*sum_i weights[i]/E(a_i) where a_i=1/(1+z*nodes[i])
    # So we are integrating f(u) = 1/E(z*u) over u in [0,1].
    
    def E_sq(t, w0, w1):
        # t is redshift z'
        # E(z)^2 = (1+z)^3 + (1+z)^(3(1+w0)) * exp(3*w1*(1 - 1/(1+z)))
        # We need to compute 1/E(z) = 1/sqrt(E_sq)
        # For numerical stability with large exponents, use logs.
        # E_sq = (1+t)^3 + (1+t)^(3+3*w0) * exp(3*w1*(t/(1+t)))
        # Note: 1 - 1/(1+t) = t/(1+t)
        
        if t == 0:
            return 1.0 # E(0) = 1
        
        log_term1 = 3.0 * np.log1p(t)
        log_term2 = (3.0 + 3.0 * w0) * np.log1p(t) + 3.0 * w1 * (t / (1.0 + t))
        
        m = np.maximum(log_term1, log_term2)
        sum_exp = np.exp(log_term1 - m) + np.exp(log_term2 - m)
        log_E_sq = m + np.log(sum_exp)
        
        return np.exp(-0.5 * log_E_sq)

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
                lambda u: E_sq(z * u, w0, w1), 
                0, 1, 
                limit=100, 
                epsabs=1e-10, 
                epsrel=1e-10
            )
            exact_ints[i] = val
        except Exception:
            us = np.linspace(0, 1, 1000)
            vals = np.array([E_sq(z * u, w0, w1) for u in us])
            exact_ints[i] = np.trapz(vals, us)
            
    if time.time() > deadline - 20:
        gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
        u0 = 0.5 + 0.5 * gl_nodes
        w0_init = 0.5 * gl_weights
        return {"nodes": u0, "weights": w0_init}

    # 3. Parameterization
    # Nodes: strictly increasing in [0,1].
    # We use a sigmoid parameterization for nodes and softmax for weights.
    
    def compute_quadrature(theta, alpha):
        u = 1.0 / (1.0 + np.exp(-np.clip(theta, -50, 50)))
        exp_alpha = np.exp(np.clip(alpha, -50, 50) - np.max(np.clip(alpha, -50, 50)))
        w = exp_alpha / np.sum(exp_alpha)
        
        t = z_arr[:, None] * u[None, :]
        
        log_term1 = 3.0 * np.log1p(t)
        log_term2 = (3.0 + 3.0 * w0_arr)[:, None] * np.log1p(t) + 3.0 * w1_arr[:, None] * (t / (1.0 + t))
        
        m = np.maximum(log_term1, log_term2)
        sum_exp = np.exp(log_term1 - m) + np.exp(log_term2 - m)
        log_E_sq = m + np.log(sum_exp)
        f_vals = np.exp(-0.5 * log_E_sq)
        
        quad_sums = np.sum(f_vals * w[None, :], axis=1)
        
        exact_ints_safe = np.maximum(exact_ints, 1e-30)
        rel_errors = np.abs(quad_sums - exact_ints_safe) / exact_ints_safe
        
        return np.max(rel_errors), u, w

    def objective(params):
        theta = params[:8]
        alpha = params[8:16]
        err, _, _ = compute_quadrature(theta, alpha)
        return err

    # 4. Initialization
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    u0 = 0.5 + 0.5 * gl_nodes
    w0_init = 0.5 * gl_weights
    
    u0_safe = np.clip(u0, 1e-9, 1 - 1e-9)
    theta0 = np.log(u0_safe / (1 - u0_safe))
    
    w0_init_safe = np.clip(w0_init, 1e-9, 1.0)
    alpha0 = np.log(w0_init_safe)
    
    p0 = np.concatenate([theta0, alpha0])
    
    best_params = p0.copy()
    best_score = objective(p0)
    
    # 5. Optimization
    # We will use a smooth approximation of the max error to allow gradient-based methods.
    # Smooth max: -log(sum(exp(-lambda * errors))) / lambda
    
    def objective_smooth(params, lam=100.0):
        theta = params[:8]
        alpha = params[8:16]
        u = 1.0 / (1.0 + np.exp(-np.clip(theta, -50, 50)))
        exp_alpha = np.exp(np.clip(alpha, -50, 50) - np.max(np.clip(alpha, -50, 50)))
        w = exp_alpha / np.sum(exp_alpha)
        
        t = z_arr[:, None] * u[None, :]
        log_term1 = 3.0 * np.log1p(t)
        log_term2 = (3.0 + 3.0 * w0_arr)[:, None] * np.log1p(t) + 3.0 * w1_arr[:, None] * (t / (1.0 + t))
        m = np.maximum(log_term1, log_term2)
        sum_exp = np.exp(log_term1 - m) + np.exp(log_term2 - m)
        log_E_sq = m + np.log(sum_exp)
        f_vals = np.exp(-0.5 * log_E_sq)
        quad_sums = np.sum(f_vals * w[None, :], axis=1)
        exact_ints_safe = np.maximum(exact_ints, 1e-30)
        rel_errors = np.abs(quad_sums - exact_ints_safe) / exact_ints_safe
        
        # Softmax approximation to max
        # To avoid overflow in exp, subtract max
        max_err = np.max(rel_errors)
        if max_err < 1e-15:
            return 0.0
        # We want to minimize the max, so we maximize -max
        # Softmax of -errors: -log(sum(exp(-lambda * errors))) / lambda
        # This is equivalent to minimizing log(sum(exp(lambda * errors))) / lambda
        # But errors are positive.
        # Let's just use a sum of powers approximation: (sum(errors^p))^(1/p)
        # For large p, this approaches max.
        p = 10.0
        return np.sum(rel_errors ** p) ** (1.0 / p)

    if time.time() < deadline - 15:
        try:
            res1 = scipy.optimize.minimize(
                objective_smooth,
                p0,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res1.fun < 1e10:
                score1 = objective(res1.x)
                if score1 < best_score:
                    best_params = res1.x.copy()
                    best_score = score1
        except Exception:
            pass

    # Polish with a direct max-error optimizer
    if time.time() < deadline - 5:
        try:
            res2 = scipy.optimize.minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res2.fun < best_score:
                best_params = res2.x.copy()
                best_score = res2.fun
        except Exception:
            pass

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
