import numpy as np
import scipy.integrate
import scipy.optimize
import time

def construct():
    start_time = time.time()
    time_limit = 85.0

    # 1. Define Design Grid
    # Dense enough to capture worst-case behavior
    w0_vals = np.linspace(-1.2, 0.5, 15)
    w1_vals = np.linspace(-1.5, 1.5, 15)
    z_vals = np.concatenate([
        np.linspace(0.01, 0.1, 5),
        np.linspace(0.2, 1.0, 10),
        np.linspace(1.5, 3.0, 10),
        np.linspace(3.5, 6.0, 5)
    ])
    
    # Meshgrid for vectorization
    W0, W1, Z = np.meshgrid(w0_vals, w1_vals, z_vals, indexing='ij')
    W0 = W0.ravel()
    W1 = W1.ravel()
    Z = Z.ravel()
    
    N_models = len(Z)
    
    # 2. Precompute Exact Integrals using Gauss-Legendre (high accuracy, fast)
    # The integrand is f(u) = exp(-log_E(z*u))
    # log_E(t) = 3(1+w0)ln(1+t) + 3w1(1 - 1/(1+t))
    
    # Use 32-point Gauss-Legendre for exact integrals
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(32)
    # Map to [0, 1]
    gl_u = 0.5 + 0.5 * gl_nodes
    gl_w = 0.5 * gl_weights
    
    # Vectorized calculation of exact integrals
    # Shape: (N_models, 32)
    t_exact = Z[:, None] * gl_u[None, :]
    log_E_exact = 3.0 * (1.0 + W0)[:, None] * np.log1p(t_exact) + 3.0 * W1[:, None] * (1.0 - 1.0 / (1.0 + t_exact))
    f_exact = np.exp(-log_E_exact)
    exact_ints = np.sum(f_exact * gl_w[None, :], axis=1)
    
    # Ensure exact_ints are positive
    exact_ints = np.maximum(exact_ints, 1e-300)

    def evaluate_rule(nodes, weights):
        """
        nodes: (8,)
        weights: (8,)
        Returns max relative error
        """
        # Sort nodes and corresponding weights
        sort_idx = np.argsort(nodes)
        nodes_sorted = nodes[sort_idx]
        weights_sorted = weights[sort_idx]
        
        # Vectorized quadrature sum
        # Shape: (N_models, 8)
        t_quad = Z[:, None] * nodes_sorted[None, :]
        log_E_quad = 3.0 * (1.0 + W0)[:, None] * np.log1p(t_quad) + 3.0 * W1[:, None] * (1.0 - 1.0 / (1.0 + t_quad))
        f_quad = np.exp(-log_E_quad)
        
        quad_sums = np.sum(f_quad * weights_sorted[None, :], axis=1)
        
        rel_errors = np.abs(quad_sums - exact_ints) / exact_ints
        return np.max(rel_errors)

    def objective(params):
        """
        params: (16,)
        First 8: logit of nodes (to map to (0,1))
        Last 8: log of weights (to map to positive, then normalize)
        """
        if time.time() - start_time > time_limit - 5:
            return np.inf
            
        node_logits = params[:8]
        weight_logs = params[8:16]
        
        # Map node logits to (0, 1) using sigmoid
        nodes = 1.0 / (1.0 + np.exp(-node_logits))
        # Clip to avoid numerical issues at boundaries
        nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
        
        # Map weight logs to positive weights and normalize
        weight_logs_max = np.max(weight_logs)
        weights_unnorm = np.exp(weight_logs - weight_logs_max)
        weights = weights_unnorm / np.sum(weights_unnorm)
        
        try:
            return evaluate_rule(nodes, weights)
        except Exception:
            return np.inf

    # 3. Optimization
    # Initial guess: Gauss-Legendre nodes and weights mapped to [0,1]
    gl8_nodes, gl8_weights = np.polynomial.legendre.leggauss(8)
    u0 = 0.5 + 0.5 * gl8_nodes
    w0 = 0.5 * gl8_weights
    
    # Convert to parameter space
    u0_safe = np.clip(u0, 1e-6, 1-1e-6)
    node_logits0 = np.log(u0_safe / (1 - u0_safe))
    w0_safe = np.clip(w0, 1e-6, 1.0)
    weight_logs0 = np.log(w0_safe)
    
    p0 = np.concatenate([node_logits0, weight_logs0])
    
    # Bounds for differential evolution
    # Node logits: [-20, 20] covers [~0, ~1]
    # Weight logs: [-20, 20] covers wide range of relative weights
    bounds = [(-20, 20)] * 8 + [(-20, 20)] * 8
    
    best_params = p0.copy()
    best_score = objective(p0)
    
    # Differential Evolution (Global Search)
    if time.time() - start_time < time_limit - 10:
        try:
            de_result = scipy.optimize.differential_evolution(
                objective, bounds, 
                seed=42, 
                maxiter=30, 
                popsize=15,
                tol=1e-4,
                mutation=(0.5, 1.0),
                recombination=0.7
            )
            if de_result.fun < best_score:
                best_params = de_result.x
                best_score = de_result.fun
        except Exception:
            pass

    # Local Refinement with Powell
    if time.time() - start_time < time_limit - 5:
        try:
            pw_result = scipy.optimize.minimize(
                objective, best_params, method='Powell',
                options={'maxiter': 200, 'xtol': 1e-10, 'ftol': 1e-12}
            )
            if pw_result.fun < best_score:
                best_params = pw_result.x
                best_score = pw_result.fun
        except Exception:
            pass
            
    # Final Nelder-Mead polish
    if time.time() - start_time < time_limit - 2:
        try:
            nm_result = scipy.optimize.minimize(
                objective, best_params, method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if nm_result.fun < best_score:
                best_params = nm_result.x
                best_score = nm_result.fun
        except Exception:
            pass

    # Extract final nodes and weights
    node_logits_final = best_params[:8]
    weight_logs_final = best_params[8:16]
    
    nodes_final = 1.0 / (1.0 + np.exp(-node_logits_final))
    nodes_final = np.clip(nodes_final, 1e-10, 1.0 - 1e-10)
    
    weight_logs_max = np.max(weight_logs_final)
    weights_final = np.exp(weight_logs_final - weight_logs_max)
    weights_final = weights_final / np.sum(weights_final)
    
    # Sort and ensure strict increase
    sort_idx = np.argsort(nodes_final)
    nodes_final = nodes_final[sort_idx]
    weights_final = weights_final[sort_idx]
    
    # Enforce strict monotonicity
    for i in range(1, 8):
        if nodes_final[i] <= nodes_final[i-1]:
            nodes_final[i] = nodes_final[i-1] + 1e-10
            
    # Re-normalize weights after sorting (though sum is invariant)
    weights_final = weights_final / np.sum(weights_final)
    
    return {"nodes": nodes_final, "weights": weights_final}
