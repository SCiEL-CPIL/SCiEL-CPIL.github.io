import numpy as np
import scipy.integrate
import scipy.optimize
import time

def construct():
    start_time = time.time()
    time_limit = 87.0  # Safety margin for 90s limit

    # Define a grid of cosmological parameters for validation
    # Focus on regions likely to produce worst-case errors
    w0_vals = np.array([
        -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5,
        -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5
    ])
    w1_vals = np.array([
        -1.5, -1.2, -0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9, 1.2, 1.5
    ])
    
    # Define a grid of redshifts - focus where errors are likely highest
    # Low z: integrand nearly constant, high z: strong variation
    z_vals = np.concatenate([
        np.linspace(0.01, 0.05, 4),
        np.linspace(0.1, 0.3, 5),
        np.linspace(0.5, 1.0, 6),
        np.linspace(1.5, 2.5, 5),
        np.linspace(3.0, 5.0, 5),
        np.linspace(5.5, 8.0, 3)
    ])
    
    # Precompute exact integrals
    exact_ints = []
    model_params = []
    
    for w0 in w0_vals:
        for w1 in w1_vals:
            for z in z_vals:
                def integrand(u, w0=w0, w1=w1, z=z):
                    t = z * u
                    log_E = 3.0 * (1.0 + w0) * np.log1p(t) + 3.0 * w1 * (1.0 - 1.0 / (1.0 + t))
                    return np.exp(-log_E)
                
                try:
                    val, _ = scipy.integrate.quad(integrand, 0, 1, limit=100, epsabs=1e-13, epsrel=1e-13)
                    exact_ints.append(val)
                    model_params.append((w0, w1, z))
                except Exception:
                    us = np.linspace(0, 1, 20001)
                    ts = z * us
                    log_E = 3.0 * (1.0 + w0) * np.log1p(ts) + 3.0 * w1 * (1.0 - 1.0 / (1.0 + ts))
                    vals = np.exp(-log_E)
                    val = np.trapz(vals, us)
                    exact_ints.append(val)
                    model_params.append((w0, w1, z))

    exact_ints = np.array(exact_ints)
    
    w0_arr = np.array([p[0] for p in model_params])
    w1_arr = np.array([p[1] for p in model_params])
    z_arr = np.array([p[2] for p in model_params])
    
    def params_to_nodes_weights(params):
        theta = params[:8]
        alpha = params[8:16]
        
        u = 1.0 / (1.0 + np.exp(-theta))
        
        alpha_max = np.max(alpha)
        exp_alpha = np.exp(alpha - alpha_max)
        w = exp_alpha / np.sum(exp_alpha)
        
        return u, w

    def objective(params):
        u, w = params_to_nodes_weights(params)
        t = z_arr[:, None] * u[None, :]
        log_E = 3.0 * (1.0 + w0_arr)[:, None] * np.log1p(t) + 3.0 * w1_arr[:, None] * (1.0 - 1.0 / (1.0 + t))
        f_vals = np.exp(-log_E)
        quad_sums = np.sum(f_vals * w[None, :], axis=1)
        rel_errors = np.abs(quad_sums - exact_ints) / exact_ints
        return np.max(rel_errors)

    def objective_smooth(params, temp=0.005):
        u, w = params_to_nodes_weights(params)
        t = z_arr[:, None] * u[None, :]
        log_E = 3.0 * (1.0 + w0_arr)[:, None] * np.log1p(t) + 3.0 * w1_arr[:, None] * (1.0 - 1.0 / (1.0 + t))
        f_vals = np.exp(-log_E)
        quad_sums = np.sum(f_vals * w[None, :], axis=1)
        rel_errors = np.abs(quad_sums - exact_ints) / exact_ints
        
        # Log-sum-exp for numerical stability
        max_err = np.max(rel_errors)
        if max_err > 0:
            scaled = (rel_errors - max_err) / temp
            sum_exp = np.sum(np.exp(scaled))
            return max_err + temp * np.log(sum_exp)
        else:
            return 0.0

    # Initial guess: Gauss-Legendre on [0,1]
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
    
    # Multi-start with different initial conditions
    initial_guesses = [p0.copy()]
    
    # Perturbed Gauss-Legendre
    for _ in range(2):
        perturbation = np.random.randn(16) * 0.1
        initial_guesses.append(p0 + perturbation)
    
    # Chebyshev-based initial guess (good for oscillatory functions)
    cheb_nodes = np.cos(np.pi * (np.arange(8) + 0.5) / 8)
    u_cheb = 0.5 + 0.5 * cheb_nodes
    u_cheb_safe = np.clip(u_cheb, 1e-9, 1 - 1e-9)
    theta_cheb = np.log(u_cheb_safe / (1 - u_cheb_safe))
    alpha_cheb = np.log(np.ones(8) / 8)
    initial_guesses.append(np.concatenate([theta_cheb, alpha_cheb]))
    
    # Radau-like (include endpoints approximately)
    u_radau = np.concatenate([[0.01, 0.99], np.linspace(0.15, 0.85, 6)])
    u_radau_safe = np.clip(u_radau, 1e-9, 1 - 1e-9)
    theta_radau = np.log(u_radau_safe / (1 - u_radau_safe))
    w_radau = np.ones(8) / 8
    alpha_radau = np.log(w_radau)
    initial_guesses.append(np.concatenate([theta_radau, alpha_radau]))
    
    # Optimize each initial guess
    for idx, p_init in enumerate(initial_guesses):
        if time.time() - start_time > time_limit - 15:
            break
        
        try:
            # Phase 1: L-BFGS-B on smooth objective
            res1 = scipy.optimize.minimize(
                objective_smooth,
                p_init,
                method='L-BFGS-B',
                options={'maxiter': 80, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 500}
            )
            
            if time.time() - start_time > time_limit - 10:
                break
            
            if objective(res1.x) < best_score:
                best_params = res1.x.copy()
                best_score = objective(res1.x)
            
            # Phase 2: Powell refinement on raw objective
            if time.time() - start_time < time_limit - 8:
                res2 = scipy.optimize.minimize(
                    objective,
                    best_params,
                    method='Powell',
                    options={'maxiter': 100, 'xtol': 1e-10, 'ftol': 1e-13}
                )
                
                if res2.fun < best_score:
                    best_params = res2.x.copy()
                    best_score = res2.fun
                    
        except Exception:
            continue
    
    # Final refinement if time permits
    if time.time() - start_time < time_limit - 5:
        try:
            res_final = scipy.optimize.minimize(
                objective,
                best_params,
                method='Powell',
                options={'maxiter': 50, 'xtol': 1e-11, 'ftol': 1e-14}
            )
            if res_final.fun < best_score:
                best_params = res_final.x.copy()
                best_score = res_final.fun
        except Exception:
            pass
    
    u_final, w_final = params_to_nodes_weights(best_params)
    
    # Ensure strict increasing order
    sort_idx = np.argsort(u_final)
    u_sorted = u_final[sort_idx]
    w_sorted = w_final[sort_idx]
    
    # Normalize weights
    w_norm = w_sorted / np.sum(w_sorted)
    
    return {"nodes": u_sorted, "weights": w_norm}
