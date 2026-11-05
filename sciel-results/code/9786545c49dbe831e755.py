"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = 85.0  # 5s safety margin before 90s limit
    
    def E_of_a(a, w0, wa):
        """H(a)/H0 for CPL model with Omega_m=0.3, Omega_de=0.7"""
        a_safe = np.maximum(a, 1e-10)
        omega_m = 0.3
        omega_de = 0.7
        # E^2 = Omega_m * a^{-3} + Omega_de * a^{3(1+w0+wa)} * exp(3*wa*(1-a))
        term_m = omega_m * a_safe**(-3)
        term_de = omega_de * a_safe**(3*(1+w0+wa)) * np.exp(3*wa*(1-a_safe))
        return np.sqrt(term_m + term_de)
    
    def reference_integral(z, w0, wa, n_points=2000):
        """High-accuracy reference using dense sampling"""
        t_arr = np.linspace(0, 1, n_points)
        a_arr = 1.0 / (1.0 + z * t_arr)
        f_arr = 1.0 / E_of_a(a_arr, w0, wa)
        return np.trapz(f_arr, t_arr)
    
    def quadrature_value(z, nodes, weights, w0, wa):
        """Compute quadrature: (1+z) * z * sum_i w_i / E(a_i)"""
        a_i = 1.0 / (1.0 + z * nodes)
        E_vals = E_of_a(a_i, w0, wa)
        return (1.0 + z) * z * np.sum(weights / E_vals)
    
    def worst_case_error(nodes, weights, z_list, model_list):
        """Compute worst relative error across all test (z, model) pairs"""
        max_err = 0.0
        for z in z_list:
            for w0, wa in model_list:
                ref = reference_integral(z, w0, wa, n_points=1000)
                quad = quadrature_value(z, nodes, weights, w0, wa)
                if abs(ref) > 1e-15:
                    rel_err = abs(quad - ref) / abs(ref) * 100.0
                    max_err = max(max_err, rel_err)
        return max_err
    
    # Define test cosmologies (CPL parameters) and redshifts
    model_list = [
        (0.0, 0.0),    # LCDM-like
        (-1.0, 0.0),   # Cosmological constant
        (-0.5, -0.5),  # Quintessence
        (-1.5, 0.5),   # Phantom
        (-0.9, 0.3),   # Mildly varying
        (-1.1, -0.2),  # Another variation
        (-0.7, 0.7),   # Strongly varying
        (-1.3, 0.1),   # Phantom mild
    ]
    z_list = [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0]
    
    # Simple objective: just worst-case error
    def objective(params):
        if time.time() > deadline - 5:
            return 1e10
        
        # Unpack and decode nodes and weights
        nodes = params[:8]
        log_weights = params[8:16]
        
        # Ensure nodes are strictly increasing in (0,1)
        # Use softplus to ensure positivity, then sort
        nodes = 0.5 * (nodes + 1.0)  # Map from [-1,1] to [0,1]
        nodes = np.clip(nodes, 0.001, 0.999)
        nodes = np.sort(nodes)
        
        # Ensure strict increase by adding small offsets
        for i in range(1, 8):
            if nodes[i] <= nodes[i-1]:
                nodes[i] = nodes[i-1] + 0.0001
        
        # Weights via softmax
        exp_w = np.exp(log_weights - np.max(log_weights))
        weights = exp_w / np.sum(exp_w)
        
        return worst_case_error(nodes, weights, z_list, model_list)
    
    # Initial parameters from Gauss-Legendre
    from numpy.polynomial.legendre import leggauss
    nodes_init, weights_init = leggauss(8)
    nodes_init = 2.0 * nodes_init - 1.0  # Map from [0,1] to [-1,1]
    
    log_weights_init = np.log(weights_init)
    params_init = np.concatenate([nodes_init, log_weights_init])
    
    best_params = params_init.copy()
    best_score = objective(params_init)
    
    try:
        # Use Nelder-Mead for local optimization
        result = minimize(
            objective,
            best_params,
            method='Nelder-Mead',
            options={
                'maxiter': 500,
                'xatol': 1e-10,
                'fatol': 1e-12,
                'adaptive': True
            }
        )
        
        if result.fun < best_score:
            best_params = result.x
            best_score = result.fun
        
        # Try a few random restarts if time permits
        rng = np.random.RandomState(42)
        for restart in range(3):
            if time.time() > deadline - 20:
                break
            
            # Random initialization near the best solution
            perturbation = rng.randn(16) * 0.1
            params_trial = best_params + perturbation
            
            result_trial = minimize(
                objective,
                params_trial,
                method='Nelder-Mead',
                options={
                    'maxiter': 200,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            
            if result_trial.fun < best_score:
                best_params = result_trial.x
                best_score = result_trial.fun
                
    except Exception as e:
        pass
    
    # Extract best nodes and weights
    nodes = best_params[:8]
    log_weights = best_params[8:16]
    
    # Decode nodes
    nodes = 0.5 * (nodes + 1.0)  # Map from [-1,1] to [0,1]
    nodes = np.clip(nodes, 0.001, 0.999)
    nodes = np.sort(nodes)
    
    # Ensure strict increase
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 0.0001
    
    # Decode weights
    exp_w = np.exp(log_weights - np.max(log_weights))
    weights = exp_w / np.sum(exp_w)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
