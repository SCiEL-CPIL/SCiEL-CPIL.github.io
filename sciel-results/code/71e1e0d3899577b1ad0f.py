"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = 85.0  # 5s safety margin before 90s limit
    
    def E_of_a(a, w0, wa):
        """H(a)/H0 for CPL model: E^2 = a^(-3(1+w0+wa)) * a^(-3wa) = a^(-3-3w0-3wa+3wa) = a^(-3(1+w0))"""
        # E^2(a) = (a/a0)^(-3(1+w0+wa)) * (a/a0)^(-3wa) where a0=1
        # For CPL: w(a) = w0 + wa*(1-a), E^2 = a^{-3(1+w0+wa)} * a^{-3wa}
        # Actually: E^2 = (1+z)^3 * (1+z)^{3wa} * a^{-3wa} ... let me use standard form
        # E^2(a) = a^{-3(1+w0)} * a^{-3wa} * a^{3wa} ... 
        # Standard CPL: w(a) = w0 + wa*(1-a)
        # E^2(a) = a^{-3(1+w0+wa)} * (1/a)^{-3wa} ... 
        # Let me use: E^2 = (a/1)^{-3(1+w0+wa)} * (a/1)^{-3wa} is wrong
        # Correct: E^2(a) = a^{-3(1+w0+wa)} * (1+a... no
        # For flat universe: E^2 = Omega_m * a^{-3} + Omega_de * exp(3*int_1^a (1+w(a'))/a' da')
        # With w(a) = w0 + wa*(1-a): int (1+w)/a da = (1+w0)*ln(a) + wa*(1-a) + wa*ln(a)
        # = (1+w0+wa)*ln(a) + wa*(1-a)
        # So E^2 = Omega_m * a^{-3} + Omega_de * a^{3(1+w0+wa)} * exp(3*wa*(1-a))
        # For simplicity in optimization, use a simplified but representative form
        
        # Actually for the quadrature task, the evaluator computes E(a) itself.
        # I need to define E(a) for my optimization. Let me use the standard CPL form.
        a_safe = np.maximum(a, 1e-10)
        omega_m = 0.3
        omega_de = 0.7
        # E^2 = Omega_m * a^{-3} + Omega_de * a^{3(1+w0+wa)} * exp(3*wa*(1-a))
        term_m = omega_m * a_safe**(-3)
        term_de = omega_de * a_safe**(3*(1+w0+wa)) * np.exp(3*wa*(1-a_safe))
        return np.sqrt(term_m + term_de)
    
    def integrand(z, w0, wa):
        """The integrand for dL/(c/H0) = integral_0^z (1+z')/E(a(z')) dz' where a=1/(1+z)"""
        a = 1.0 / (1.0 + z)
        return (1.0 + z) / E_of_a(a, w0, wa)
    
    def reference_integral(z, w0, wa, n_points=1000):
        """High-accuracy reference using dense sampling"""
        z_arr = np.linspace(0, z, n_points)
        f = integrand(z_arr, w0, wa)
        # Use trapezoidal rule with many points
        return np.trapz(f, z_arr)
    
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
                ref = reference_integral(z, w0, wa, n_points=500)
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
    
    # Initial guess: use Gauss-Legendre nodes for better starting point
    from numpy.polynomial.legendre import leggauss
    nodes_init, _ = leggauss(8)
    nodes_init = 0.5 * (nodes_init + 1.0)  # Map from [-1,1] to [0,1]
    weights_init = np.ones(8) / 8.0
    
    # Optimize nodes and weights jointly
    # Parameters: 8 nodes (constrained to be in (0,1) and increasing) + 8 weights (positive, sum=1)
    # Use parameterization: nodes via tanh to keep in (0,1), weights via softmax-like
    
    def param_to_nodes_weights(params):
        """Convert unconstrained params to valid nodes and weights"""
        node_params = params[:8]
        weight_params = params[8:16]
        
        # Map node_params to (0,1) using sigmoid, then enforce ordering
        raw_nodes = 1.0 / (1.0 + np.exp(-node_params))
        # Enforce strict ordering by sorting and adding small offsets
        sorted_nodes = np.sort(raw_nodes)
        # Add small offsets to ensure strict increase
        nodes = sorted_nodes + np.arange(8) * 0.001
        # Clip to (0,1)
        nodes = np.clip(nodes, 0.01, 0.99)
        
        # Weights: use softmax to ensure positivity and sum to 1
        exp_w = np.exp(weight_params - np.max(weight_params))
        weights = exp_w / np.sum(exp_w)
        
        return nodes, weights
    
    def objective(params):
        if time.time() > deadline - 5:
            return 1e10  # Signal timeout
        nodes, weights = param_to_nodes_weights(params)
        return worst_case_error(nodes, weights, z_list, model_list)
    
    # Initial parameters
    node_params_init = np.zeros(8)
    weight_params_init = np.zeros(8)
    params_init = np.concatenate([node_params_init, weight_params_init])
    
    # Run optimization with time guard
    best_params = params_init.copy()
    best_score = objective(params_init)
    
    try:
        result = minimize(
            objective,
            params_init,
            method='Nelder-Mead',
            options={
                'maxiter': 500,
                'xatol': 1e-6,
                'fatol': 1e-8,
                'adaptive': True
            }
        )
        
        if time.time() < deadline - 10:
            # Try another method if time permits
            result2 = minimize(
                objective,
                result.x,
                method='Powell',
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            if result2.fun < result.fun:
                result = result2
        
        if result.fun < best_score:
            best_params = result.x
            best_score = result.fun
            
    except Exception as e:
        # Fall back to initial or best found
        pass
    
    # Extract best nodes and weights
    nodes, weights = param_to_nodes_weights(best_params)
    
    # Ensure nodes are strictly increasing and in (0,1)
    nodes = np.sort(nodes)
    nodes = np.clip(nodes, 0.01, 0.99)
    # Ensure strict increase
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 0.001
    
    # Ensure weights are positive and sum to 1
    weights = np.abs(weights)
    weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
