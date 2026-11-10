"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    deadline = time.time() + 85  # 85s deadline, 5s safety margin
    
    def E_of_a(a, w0, wa):
        """CPL dark energy model: E(a) = H(a)/H0"""
        a_safe = np.maximum(a, 1e-15)
        return a_safe**(-3.0*wa/2.0) * (1.0 - a_safe)**(1.5*(1.0 + w0 + wa))
    
    def luminosity_distance(nodes, weights, w0, wa, z):
        """Compute dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)"""
        a_i = 1.0 / (1.0 + z * nodes)
        E_vals = E_of_a(a_i, w0, wa)
        return (1.0 + z) * z * np.sum(weights / E_vals)
    
    def reference_integral(w0, wa, z, n_points=10000):
        """High-accuracy reference using adaptive quadrature"""
        from scipy.integrate import quad
        def integrand(zp):
            a = 1.0 / (1.0 + zp)
            return 1.0 / E_of_a(a, w0, wa)
        result, _ = quad(integrand, 0, z, limit=200)
        return (1.0 + z) * z * result
    
    def objective(params):
        """Minimize worst-case relative error (smooth surrogate)"""
        # Unpack parameters: 8 nodes + 8 weights
        nodes = params[:8]
        weights = params[8:]
        
        # Enforce constraints softly
        # Nodes must be in (0,1) and strictly increasing
        node_violation = 0.0
        for i in range(7):
            if nodes[i+1] - nodes[i] < 0.01:
                node_violation += (0.01 - (nodes[i+1] - nodes[i]))**2
        if nodes[0] < 0.01:
            node_violation += (0.01 - nodes[0])**2
        if nodes[7] > 0.99:
            node_violation += (nodes[7] - 0.99)**2
        
        # Weights must be positive
        weight_violation = 0.0
        for w in weights:
            if w < 0.001:
                weight_violation += (0.001 - w)**2
        
        # Weight sum should be 1
        weight_sum_violation = (np.sum(weights) - 1.0)**2
        
        # Compute errors across representative cosmologies
        w0_vals = [-1.0, -0.8, -0.5, 0.0, 0.5]
        wa_vals = [-0.5, 0.0, 0.5]
        z_vals = [0.5, 1.0, 2.0, 3.0]
        
        errors = []
        for w0 in w0_vals:
            for wa in wa_vals:
                for z in z_vals:
                    dl_quad = luminosity_distance(nodes, weights, w0, wa, z)
                    dl_ref = reference_integral(w0, wa, z)
                    if abs(dl_ref) > 1e-10:
                        rel_err = abs(dl_quad - dl_ref) / abs(dl_ref)
                        errors.append(rel_err)
        
        if len(errors) == 0:
            return 1e6
        
        errors = np.array(errors)
        # Soft max: log-sum-exp for smooth worst-case
        max_err = np.max(errors)
        soft_max = np.log(np.sum(np.exp(100 * (errors - max_err)))) / 100 + max_err
        
        return soft_max + 1000 * (node_violation + weight_violation + weight_sum_violation)
    
    # Initial guess: Gauss-Legendre nodes and weights
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    # Map from [-1,1] to [0,1]
    gl_nodes = (gl_nodes + 1.0) / 2.0
    gl_nodes = np.sort(gl_nodes)
    
    x0 = np.concatenate([gl_nodes, gl_weights])
    
    # Optimize
    best_params = x0
    best_obj = objective(x0)
    
    result = minimize(objective, x0, method='L-BFGS-B', 
                      options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
    
    if result.fun < best_obj:
        best_params = result.x
        best_obj = result.fun
    
    # Check time
    if time.time() > deadline - 10:
        params = best_params
    else:
        # Try a few more optimizations with different initial guesses
        for _ in range(5):
            if time.time() > deadline - 15:
                break
            # Perturb initial guess
            perturbation = np.random.randn(16) * 0.01
            x_trial = x0 + perturbation
            # Ensure basic validity
            x_trial[:8] = np.clip(np.sort(x_trial[:8]), 0.01, 0.99)
            x_trial[8:] = np.clip(x_trial[8:], 0.001, None)
            x_trial[8:] = x_trial[8:] / np.sum(x_trial[8:])
            
            res = minimize(objective, x_trial, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x
        
        params = best_params
    
    # Extract and validate
    nodes = params[:8]
    weights = params[8:]
    
    # Ensure strict increasing and in [0,1]
    nodes = np.sort(nodes)
    nodes = np.clip(nodes, 0.001, 0.999)
    # Ensure minimum spacing
    for i in range(7):
        if nodes[i+1] - nodes[i] < 0.001:
            nodes[i+1] = nodes[i] + 0.001
    nodes[7] = min(nodes[7], 0.999)
    
    # Ensure positive weights summing to 1
    weights = np.maximum(weights, 1e-6)
    weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
