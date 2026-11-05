"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
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
    
    def integrand(z, w0, wa):
        """The integrand for dL/(c/H0) = integral_0^z (1+z')/E(a(z')) dz'"""
        a = 1.0 / (1.0 + z)
        return (1.0 + z) / E_of_a(a, w0, wa)
    
    def reference_integral(z, w0, wa, n_points=1000):
        """High-accuracy reference using dense sampling"""
        z_arr = np.linspace(0, z, n_points)
        f = integrand(z_arr, w0, wa)
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
    
    # Moment-matching strategy: choose basis functions that span the integrand family
    # The integrand is (1+z)/E(a(z)) where a=1/(1+z)
    # For various w0,wa, this function has different curvature
    # Use Chebyshev polynomials on [0,1] as basis for moment matching
    
    def chebyshev_basis(x, n_max=7):
        """Evaluate Chebyshev polynomials T_0 through T_n_max at x"""
        basis = np.zeros((len(x), n_max+1))
        basis[:, 0] = 1.0
        if n_max >= 1:
            basis[:, 1] = x
        for k in range(2, n_max+1):
            basis[:, k] = 2*x*basis[:, k-1] - basis[:, k-2]
        return basis
    
    # Select representative models for moment matching
    moment_models = [
        (0.0, 0.0),
        (-1.0, 0.0),
        (-0.5, -0.5),
        (-1.5, 0.5),
        (-0.9, 0.3),
    ]
    moment_z = 1.0  # Reference redshift for moment matching
    
    def moment_residuals(params):
        """Residuals for moment matching: quadrature should integrate basis functions exactly"""
        node_params = params[:8]
        weight_params = params[8:16]
        
        # Map node_params to (0,1) using sigmoid
        raw_nodes = 1.0 / (1.0 + np.exp(-node_params))
        sorted_nodes = np.sort(raw_nodes)
        nodes = sorted_nodes + np.arange(8) * 0.001
        nodes = np.clip(nodes, 0.01, 0.99)
        
        # Weights via softmax
        exp_w = np.exp(weight_params - np.max(weight_params))
        weights = exp_w / np.sum(exp_w)
        
        # For moment matching, we want the quadrature to integrate
        # f(z) = (1+z)/E(a(z)) * basis_k(z) for each basis function
        # But since E depends on the model, we use a representative set
        
        residuals = []
        for w0, wa in moment_models:
            for k in range(4):  # Match first 4 moments
                # Compute reference integral of (1+z)/E(a) * T_k(z) on [0, moment_z]
                z_arr = np.linspace(0, moment_z, 500)
                a_arr = 1.0 / (1.0 + z_arr)
                f_arr = (1.0 + z_arr) / E_of_a(a_arr, w0, wa)
                basis_vals = chebyshev_basis(z_arr, 3)[:, k]
                ref_moment = np.trapz(f_arr * basis_vals, z_arr)
                
                # Quadrature approximation
                a_i = 1.0 / (1.0 + moment_z * nodes)
                f_i = (1.0 + moment_z) / E_of_a(a_i, w0, wa)
                basis_i = chebyshev_basis(nodes, 3)[:, k]
                quad_moment = moment_z * np.sum(weights * f_i * basis_i)
                
                residuals.append((quad_moment - ref_moment) / max(abs(ref_moment), 1e-10))
        
        return residuals
    
    # Combined objective: moment matching + worst-case error
    def combined_objective(params):
        if time.time() > deadline - 5:
            return 1e10
        
        node_params = params[:8]
        weight_params = params[8:16]
        
        raw_nodes = 1.0 / (1.0 + np.exp(-node_params))
        sorted_nodes = np.sort(raw_nodes)
        nodes = sorted_nodes + np.arange(8) * 0.001
        nodes = np.clip(nodes, 0.01, 0.99)
        
        exp_w = np.exp(weight_params - np.max(weight_params))
        weights = exp_w / np.sum(exp_w)
        
        # Worst-case error
        wc_err = worst_case_error(nodes, weights, z_list, model_list)
        
        # Moment matching penalty
        res = moment_residuals(params)
        moment_penalty = np.max(np.abs(res)) if len(res) > 0 else 0
        
        # Combine: prioritize worst-case error, use moments as regularizer
        return wc_err + 0.1 * moment_penalty
    
    # Initial parameters from Gauss-Legendre
    from numpy.polynomial.legendre import leggauss
    nodes_init, weights_init = leggauss(8)
    nodes_init = 0.5 * (nodes_init + 1.0)
    
    # Invert sigmoid to get initial node_params
    node_params_init = np.log(nodes_init / (1.0 - nodes_init))
    weight_params_init = np.log(weights_init)
    params_init = np.concatenate([node_params_init, weight_params_init])
    
    # Run differential evolution for global search
    best_params = params_init.copy()
    best_score = combined_objective(params_init)
    
    try:
        # Differential evolution for global exploration
        bounds = [(-3, 3)] * 8 + [(-2, 2)] * 8
        result = differential_evolution(
            combined_objective,
            bounds,
            maxiter=50,
            tol=1e-6,
            popsize=15,
            seed=42,
            polish=False
        )
        
        if result.fun < best_score:
            best_params = result.x
            best_score = result.fun
        
        # Local refinement with Nelder-Mead
        if time.time() < deadline - 15:
            result2 = minimize(
                combined_objective,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 300,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            if result2.fun < best_score:
                best_params = result2.x
                best_score = result2.fun
        
        # Final polish with Powell if time permits
        if time.time() < deadline - 10:
            result3 = minimize(
                combined_objective,
                best_params,
                method='Powell',
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            if result3.fun < best_score:
                best_params = result3.x
                best_score = result3.fun
                
    except Exception as e:
        pass
    
    # Extract best nodes and weights
    node_params = best_params[:8]
    weight_params = best_params[8:16]
    
    raw_nodes = 1.0 / (1.0 + np.exp(-node_params))
    nodes = np.sort(raw_nodes)
    nodes = nodes + np.arange(8) * 0.001
    nodes = np.clip(nodes, 0.01, 0.99)
    
    # Ensure strict increase
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 0.001
    
    exp_w = np.exp(weight_params - np.max(weight_params))
    weights = exp_w / np.sum(exp_w)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
