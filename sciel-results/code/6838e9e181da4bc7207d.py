"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def construct():
    deadline = time.time() + 80.0
    
    def E_of_a(a, w0, wa, wb):
        """E(a) = H(a)/H0 for CPL dark energy with w0, wa, wb (quadratic CPL)."""
        # E^2 = (1-a)^(-2(1+w0+wa+wb/2)) * a^(-3(1+w0+wa+wb)) 
        # More precisely for CPL: E^2 = (1+z)^(3(1+w0)) * (1+w0+wa*z)^(3*wa/(1+w0))
        # In terms of a: 1+z = 1/a
        # E^2 = a^(-3(1+w0)) * (1 + wa*(1/a - 1))^(3*wa/(1+w0))
        # Simplify: 1 + wa*(1/a - 1) = (1/a)(1 + wa*a - wa) = (1/a)(1 - wa + wa*a)
        # E^2 = a^(-3(1+w0)) * [(1 - wa + wa*a)/a]^(3*wa/(1+w0))
        # E^2 = a^(-3(1+w0)) * (1 - wa + wa*a)^(3*wa/(1+w0)) * a^(-3*wa/(1+w0))
        # E^2 = a^(-3(1+w0) - 3*wa/(1+w0)) * (1 - wa + wa*a)^(3*wa/(1+w0))
        # For general wb, we can use a more general form. Let's use the standard CPL with only w0, wa.
        # E^2 = a^(-3(1+w0)) * (1 - wa + wa*a)^(3*wa/(1+w0))
        
        exponent_a = -3.0 * (1.0 + w0)
        exponent_cpl = 3.0 * wa / (1.0 + w0)
        
        cpl_term = (1.0 - wa + wa * a)
        # Handle potential negative base with fractional exponent
        if cpl_term <= 0:
            return 1e6
        e2 = (a ** exponent_a) * (cpl_term ** exponent_cpl)
        if e2 <= 0:
            return 1e6
        return np.sqrt(e2)
    
    def integrand(z, nodes, w0, wa, wb):
        """(1+z) * z / E(a) where a = 1/(1+z*node)"""
        a = 1.0 / (1.0 + z * nodes)
        e_vals = np.array([E_of_a(ai, w0, wa, wb) for ai in a])
        return (1.0 + z) * z / e_vals
    
    def luminosity_distance(z, nodes, weights, w0, wa, wb):
        """Compute dL/(c/H0) = (1+z) * z * sum_i weights[i] / E(a_i)"""
        if z < 1e-10:
            return 0.0
        a = 1.0 / (1.0 + z * nodes)
        e_vals = np.array([E_of_a(ai, w0, wa, wb) for ai in a])
        return (1.0 + z) * z * np.sum(weights / e_vals)
    
    def reference_distance(z, w0, wa, wb, n_steps=10000):
        """High-accuracy adaptive integration for reference."""
        if z < 1e-10:
            return 0.0
        # Use trapezoidal rule with many steps in z-space
        z_arr = np.linspace(0, z, n_steps)
        a_arr = 1.0 / (1.0 + z_arr)
        e_arr = np.array([E_of_a(ai, w0, wa, wb) for ai in a_arr])
        integrand_arr = z_arr / e_arr
        # Trapezoidal integration
        integral = np.trapz(integrand_arr, z_arr)
        return (1.0 + z) * integral
    
    def quadrature_error(z, nodes, weights, w0, wa, wb):
        """Relative error for a given z."""
        quad_val = luminosity_distance(z, nodes, weights, w0, wa, wb)
        ref_val = reference_distance(z, w0, wa, wb)
        if abs(ref_val) < 1e-15:
            return 0.0
        return abs(quad_val - ref_val) / abs(ref_val)
    
    def objective(params):
        """Minimize worst-case relative error across test cosmologies."""
        nodes = params[:8]
        log_w = params[8:16]
        weights = np.exp(log_w)
        weights = weights / np.sum(weights)
        
        # Ensure nodes are strictly increasing
        nodes = np.sort(nodes)
        nodes[0] = max(nodes[0], 1e-4)
        nodes[-1] = min(nodes[-1], 1.0 - 1e-4)
        for i in range(7):
            if nodes[i+1] - nodes[i] < 1e-6:
                nodes[i+1] = nodes[i] + 1e-5
        
        # Test on several cosmologies and redshifts
        # w0, wa pairs: (0,0), (-1,0), (-0.5,0), (-1,0.5), (-0.8,0.3), (-1.2,-0.2)
        test_models = [
            (0.0, 0.0),      # Lambda CDM
            (-1.0, 0.0),     # Cosmological constant
            (-0.5, 0.0),     # Quintessence
            (-1.0, 0.5),     # CPL with positive wa
            (-0.8, 0.3),     # CPL
            (-1.2, -0.2),    # CPL with negative wa
        ]
        
        z_tests = [0.1, 0.2, 0.5, 1.0, 2.0]
        
        max_err = 0.0
        for w0, wa in test_models:
            for z in z_tests:
                err = quadrature_error(z, nodes, weights, w0, wa, 0.0)
                if err > max_err:
                    max_err = err
        return max_err
    
    # Initial guess: equally spaced nodes, equal weights
    n = 8
    nodes_init = (np.arange(n) + 0.5) / n
    log_w_init = np.zeros(n)
    x0 = np.concatenate([nodes_init, log_w_init])
    
    best_score = objective(x0)
    best_params = x0.copy()
    
    # Multi-restart Nelder-Mead optimization
    np.random.seed(42)
    for restart in range(20):
        if time.time() > deadline - 15:
            break
        
        if restart == 0:
            params_init = x0.copy()
        else:
            nodes_pert = x0[:8] + np.random.randn(8) * 0.03
            nodes_pert = np.sort(nodes_pert)
            nodes_pert[0] = max(nodes_pert[0], 0.01)
            nodes_pert[-1] = min(nodes_pert[-1], 0.99)
            for i in range(7):
                if nodes_pert[i+1] - nodes_pert[i] < 0.005:
                    nodes_pert[i+1] = nodes_pert[i] + 0.005
            if nodes_pert[-1] >= 1.0:
                nodes_pert = np.linspace(0.05, 0.95, 8)
            logw_pert = np.random.randn(8) * 0.3
            params_init = np.concatenate([nodes_pert, logw_pert])
        
        try:
            result = minimize(objective, params_init, method='Nelder-Mead',
                              options={'maxiter': 200, 'xatol': 1e-9, 'fatol': 1e-10, 'adaptive': True})
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Extract best nodes and weights
    best_nodes = np.sort(best_params[:8].copy())
    best_logw = best_params[8:16].copy()
    best_w = np.exp(best_logw)
    best_w = best_w / np.sum(best_w)
    
    # Ensure nodes are strictly increasing in (0,1)
    best_nodes[0] = max(best_nodes[0], 1e-6)
    best_nodes[-1] = min(best_nodes[-1], 1.0 - 1e-6)
    for i in range(7):
        if best_nodes[i+1] - best_nodes[i] < 1e-8:
            best_nodes[i+1] = best_nodes[i] + 1e-6
    
    # Final normalization
    best_w = best_w / np.sum(best_w)
    
    return {"nodes": best_nodes, "weights": best_w}
# EVOLVE-BLOCK-END
