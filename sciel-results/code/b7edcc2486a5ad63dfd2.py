"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s deadline, 10s margin
    
    # Reference: dL/(c/H0) = (1+z) * integral_0^z du / E(a(u))
    # where a(u) = 1/(1+u), E(a) = sqrt(Om*a^-3 + Ode*a^(3*(1+w0+wa)))
    # For flat: Om + Ode = 1
    
    # The quadrature: (1+z) * sum_i w_i / E(a_i) where a_i = 1/(1+z*x_i)
    # nodes x_i in [0,1], z is the redshift
    
    # We want to minimize worst-case relative error over:
    # - z in some range (say 0.01 to 5)
    # - w0, wa in some range (say w0 in [-2,1], wa in [-1,1])
    # - Om in [0.05, 0.95]
    
    # Change of variables for constraints:
    # nodes: use logit transform: x_i = sigmoid(t_i), enforce ordering via t_i < t_{i+1}
    # weights: use log: w_i = exp(s_i), then normalize
    
    def E_of_a(a, Om, w0, wa):
        """H(a)/H0 for flat CPL cosmology"""
        Ode = 1.0 - Om
        # E^2 = Om * a^-3 + Ode * a^(3*(1+w0+wa))
        return np.sqrt(Om * a**(-3) + Ode * a**(3.0*(1.0+w0+wa)))
    
    def reference_integral(z, Om, w0, wa):
        """Compute (1+z) * integral_0^z du / E(1/(1+u)) using adaptive quadrature"""
        from scipy.integrate import quad
        def integrand(u):
            a = 1.0 / (1.0 + u)
            return 1.0 / E_of_a(a, Om, w0, wa)
        val, _ = quad(integrand, 0.0, z, limit=200)
        return (1.0 + z) * val
    
    def quadrature_value(nodes, weights, z, Om, w0, wa):
        """Compute (1+z) * sum_i w_i / E(a_i)"""
        a_vals = 1.0 / (1.0 + z * nodes)
        E_vals = E_of_a(a_vals, Om, w0, wa)
        return (1.0 + z) * np.dot(weights, 1.0 / E_vals)
    
    def relative_error(nodes, weights, z, Om, w0, wa):
        ref = reference_integral(z, Om, w0, wa)
        quad_val = quadrature_value(nodes, weights, z, Om, w0, wa)
        return abs(quad_val - ref) / ref
    
    # Optimization variables:
    # t (8): logit of nodes (nodes = sigmoid(t))
    # s (8): log of unnormalized weights (weights = exp(s)/sum(exp(s)))
    
    def params_to_nodes_weights(t, s):
        nodes = 1.0 / (1.0 + np.exp(-t))  # sigmoid
        # Enforce strict ordering by adding small offsets
        # Actually, sigmoid is monotonic, so if t is sorted, nodes are sorted
        # But we need to ensure t is sorted - handle in objective
        w_unnorm = np.exp(s)
        weights = w_unnorm / np.sum(w_unnorm)
        return nodes, weights
    
    def objective(params):
        if time.time() > deadline:
            return 1e10
        t = params[:8]
        s = params[8:]
        
        nodes, weights = params_to_nodes_weights(t, s)
        
        # Check ordering
        if np.any(np.diff(nodes) <= 1e-15):
            return 1e10
        
        # Check weights positive (guaranteed by exp)
        
        # Evaluate over a grid of cosmologies and redshifts
        # Use a representative sample
        z_vals = np.array([0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
        Om_vals = np.array([0.2, 0.3, 0.5, 0.7, 0.9])
        w0_vals = np.array([-1.5, -0.5, 0.0, 0.5, 1.0])
        wa_vals = np.array([-0.5, 0.0, 0.5])
        
        # Soft-max of relative errors (log-sum-exp approximation)
        errors = []
        for z in z_vals:
            for Om in Om_vals:
                for w0 in w0_vals:
                    for wa in wa_vals:
                        if time.time() > deadline:
                            return 1e10
                        err = relative_error(nodes, weights, z, Om, w0, wa)
                        errors.append(err)
        
        errors = np.array(errors)
        # Soft max: log(sum(exp(err/tau))) * tau, with small tau
        tau = 0.01
        max_err = np.max(errors)
        soft_max = max_err + tau * np.log(np.sum(np.exp((errors - max_err) / tau)))
        
        return soft_max
    
    # Initial guess: uniform midpoint rule
    n = 8
    x0_nodes = (np.arange(n) + 0.5) / n
    x0_t = np.log(x0_nodes / (1.0 - x0_nodes))  # inverse sigmoid
    x0_s = np.zeros(n)  # equal weights
    
    x0 = np.concatenate([x0_t, x0_s])
    
    # Try L-BFGS-B
    best_params = x0.copy()
    best_obj = objective(x0)
    
    try:
        result = minimize(
            objective, x0, method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 50000}
        )
        if result.fun < best_obj:
            best_params = result.x
            best_obj = result.fun
    except Exception:
        pass
    
    # Polish with Nelder-Mead if time permits
    if time.time() < deadline - 5.0:
        try:
            result2 = minimize(
                objective, best_params, method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            if result2.fun < best_obj:
                best_params = result2.x
                best_obj = result2.fun
        except Exception:
            pass
    
    # Extract final nodes and weights
    t_final = best_params[:8]
    s_final = best_params[8:]
    
    nodes = 1.0 / (1.0 + np.exp(-t_final))
    w_unnorm = np.exp(s_final)
    weights = w_unnorm / np.sum(w_unnorm)
    
    # Ensure strict ordering and positivity
    # If nodes aren't strictly increasing, fall back to midpoint
    if np.any(np.diff(nodes) <= 1e-14) or np.any(nodes <= 0) or np.any(nodes >= 1):
        nodes = (np.arange(n) + 0.5) / n
        weights = np.full(n, 1.0 / n)
    else:
        # Clip nodes to [eps, 1-eps]
        eps = 1e-12
        nodes = np.clip(nodes, eps, 1.0 - eps)
        # Re-sort if needed (shouldn't happen)
        order = np.argsort(nodes)
        nodes = nodes[order]
        weights = weights[order]
        # Renormalize weights
        weights = weights / np.sum(weights)
    
    # Final safety: ensure weights sum to 1 and are positive
    weights = np.maximum(weights, 1e-15)
    weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}

# EVOLVE-BLOCK-END
