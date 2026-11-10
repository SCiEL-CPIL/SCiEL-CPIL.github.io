import numpy as np
import scipy.integrate
import scipy.optimize
import time

def construct():
    start_time = time.time()
    time_limit = 88.0  # Safety margin for 90s limit

    # Validation grid
    # Focus on a wide range of w0, w1 and z
    # w0 in [-1.5, 0.5], w1 in [-1.5, 1.5]
    # z in [0.01, 10.0]
    
    z_vals = np.logspace(-2, 1, 30)
    w0_vals = np.linspace(-1.5, 0.5, 8)
    w1_vals = np.linspace(-1.5, 1.5, 8)
    
    # Create meshgrid
    Z, W0, W1 = np.meshgrid(z_vals, w0_vals, w1_vals, indexing='ij')
    Z = Z.flatten()
    W0 = W0.flatten()
    W1 = W1.flatten()
    n_models = len(Z)
    
    # Precompute exact integrals
    # I = int_0^1 du / E(z*u)
    # E(t) = (1+t)^(3(1+w0)) * exp(3*w1*(1 - 1/(1+t)))
    # 1/E(t) = exp(-log_E)
    
    exact_ints = np.zeros(n_models)
    
    # Vectorized calculation of exact integrals is not directly possible with quad,
    # but we can loop. 30*8*8 = 1920 calls.
    # Each quad call is fast (~1ms). Total ~2s.
    
    for i in range(n_models):
        z = Z[i]
        w0 = W0[i]
        w1 = W1[i]
        
        def integrand(u):
            t = z * u
            # Handle t=0 carefully if needed, but log1p(0)=0
            log_E = 3.0 * (1.0 + w0) * np.log1p(t) + 3.0 * w1 * (1.0 - 1.0 / (1.0 + t))
            return np.exp(-log_E)
        
        try:
            val, _ = scipy.integrate.quad(integrand, 0, 1, limit=50, epsabs=1e-10, epsrel=1e-10)
            exact_ints[i] = val
        except Exception:
            # Fallback to simple trapezoid if quad fails (unlikely)
            us = np.linspace(0, 1, 1000)
            ts = z * us
            log_E = 3.0 * (1.0 + w0) * np.log1p(ts) + 3.0 * w1 * (1.0 - 1.0 / (1.0 + ts))
            vals = np.exp(-log_E)
            exact_ints[i] = np.trapz(vals, us)
            
    # Ensure exact_ints are positive
    exact_ints = np.maximum(exact_ints, 1e-15)

    # Parameterization
    # 16 parameters: 8 for nodes (theta), 8 for weights (alpha)
    # u_i = sigmoid(theta_i)
    # w_i = softmax(alpha_i)
    
    def params_to_nodes_weights(params):
        theta = params[:8]
        alpha = params[8:16]
        
        # Nodes
        # Use stable sigmoid
        u = 1.0 / (1.0 + np.exp(-theta))
        
        # Weights
        alpha_max = np.max(alpha)
        exp_alpha = np.exp(alpha - alpha_max)
        w = exp_alpha / np.sum(exp_alpha)
        
        return u, w

    def objective(params):
        u, w = params_to_nodes_weights(params)
        
        # Calculate quadrature sum for all models
        # t = z * u
        # Shape: (n_models, 8)
        t = Z[:, None] * u[None, :]
        
        # Calculate 1/E(t)
        # log_E = 3(1+w0)log(1+t) + 3w1(1 - 1/(1+t))
        # w0, w1 shape: (n_models,)
        log_E = 3.0 * (1.0 + W0)[:, None] * np.log1p(t) + 3.0 * W1[:, None] * (1.0 - 1.0 / (1.0 + t))
        f_vals = np.exp(-log_E)
        
        # Quadrature sum: sum_i w_i f(t_i)
        quad_sums = np.sum(f_vals * w[None, :], axis=1)
        
        # Relative error
        rel_errors = np.abs(quad_sums - exact_ints) / exact_ints
        
        # Return max relative error
        return np.max(rel_errors)
        
    def objective_smooth(params):
        # Soft max for differentiability
        u, w = params_to_nodes_weights(params)
        t = Z[:, None] * u[None, :]
        log_E = 3.0 * (1.0 + W0)[:, None] * np.log1p(t) + 3.0 * W1[:, None] * (1.0 - 1.0 / (1.0 + t))
        f_vals = np.exp(-log_E)
        quad_sums = np.sum(f_vals * w[None, :], axis=1)
        rel_errors = np.abs(quad_sums - exact_ints) / exact_ints
        
        # Temperature for soft max
        temp = 0.005
        # log(sum(exp(err/temp))) * temp
        # To avoid overflow, subtract max
        max_err = np.max(rel_errors)
        return np.log(np.sum(np.exp((rel_errors - max_err) / temp))) * temp + max_err

    # Initial guess: Gauss-Legendre nodes and weights on [0,1]
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    u0 = 0.5 + 0.5 * gl_nodes
    w0_init = 0.5 * gl_weights
    
    # Inverse transform for parameters
    u0_safe = np.clip(u0, 1e-9, 1-1e-9)
    theta0 = np.log(u0_safe / (1 - u0_safe))
    
    w0_init_safe = np.clip(w0_init, 1e-9, 1.0)
    alpha0 = np.log(w0_init_safe)
    
    p0 = np.concatenate([theta0, alpha0])
    
    best_params = p0
    best_score = objective(p0)
    
    # Optimization Step 1: L-BFGS-B on smooth objective
    try:
        res1 = scipy.optimize.minimize(
            objective_smooth, 
            p0, 
            method='L-BFGS-B', 
            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-8}
        )
        
        if time.time() - start_time < time_limit - 10:
            score1 = objective(res1.x)
            if score1 < best_score:
                best_params = res1.x
                best_score = score1
    except Exception:
        pass

    # Optimization Step 2: Powell on raw objective
    try:
        res2 = scipy.optimize.minimize(
            objective, 
            best_params, 
            method='Powell', 
            options={'maxiter': 300, 'xtol': 1e-10, 'ftol': 1e-12}
        )
        
        if time.time() - start_time < time_limit - 5:
            score2 = objective(res2.x)
            if score2 < best_score:
                best_params = res2.x
                best_score = score2
    except Exception:
        pass

    # Optimization Step 3: Further refinement with Nelder-Mead if time allows
    if time.time() - start_time < time_limit - 2:
        try:
            res3 = scipy.optimize.minimize(
                objective, 
                best_params, 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            score3 = objective(res3.x)
            if score3 < best_score:
                best_params = res3.x
                best_score = score3
        except Exception:
            pass

    u_final, w_final = params_to_nodes_weights(best_params)
    
    # Ensure strict increasing order of nodes
    sort_idx = np.argsort(u_final)
    u_sorted = u_final[sort_idx]
    w_sorted = w_final[sort_idx]
    
    # Normalize weights to sum to 1 exactly
    w_norm = w_sorted / np.sum(w_sorted)
    
    # Final check: ensure nodes are in (0,1)
    # Sigmoid ensures this, but clip just in case
    u_sorted = np.clip(u_sorted, 1e-12, 1-1e-12)
    
    return {"nodes": u_sorted, "weights": w_norm}
