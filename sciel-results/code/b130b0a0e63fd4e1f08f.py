"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def construct():
    """
    Construct an 8-node positive quadrature rule for dimensionless luminosity distance.
    
    Strategy: Start from Gauss-Legendre (optimal for polynomials) and refine
    to minimize worst-case relative error across diverse CPL cosmologies.
    Use a robust parameterization and minimize the maximum error (Chebyshev criterion).
    """
    deadline = time.time() + 85  # 85s deadline, 5s safety margin
    
    # Representative CPL cosmologies: diverse (w0, wa) pairs
    # Include phantom (w < -1), cosmological constant (w = -1), quintessence (-1 < w < -0.5)
    w0_values = [-1.5, -1.2, -1.0, -0.8, -0.5, 0.0, 0.5]
    wa_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    cosmologies = []
    for w0 in w0_values:
        for wa in wa_values:
            cosmologies.append((w0, wa))
    
    Om_m = 0.3
    Om_de = 0.7
    
    def E_of_a(a, w0, wa):
        """Compute E(a) = H(a)/H0 for CPL model."""
        a = np.asarray(a)
        matter = Om_m / a**3
        if abs(w0 + wa) < 1e-10:
            de_term = Om_de * a**(-3*(1+w0))
        else:
            exponent = 3*wa/(1+w0+wa)
            de_term = Om_de * a**(-3*(1+w0+wa)) * (1 + wa*(1-a))**exponent
        return np.sqrt(matter + de_term)
    
    def reference_integral(z, w0, wa, n_ref=5000):
        """High-accuracy reference integral using Simpson's rule."""
        u = np.linspace(0, 1, n_ref)
        a_vals = 1.0 / (1.0 + z * u)
        E_vals = E_of_a(a_vals, w0, wa)
        integrand = 1.0 / E_vals
        
        # Ensure odd number of points for Simpson's rule
        if n_ref % 2 == 0:
            n_ref = n_ref - 1
            u = np.linspace(0, 1, n_ref)
            a_vals = 1.0 / (1.0 + z * u)
            E_vals = E_of_a(a_vals, w0, wa)
            integrand = 1.0 / E_vals
        
        h = 1.0 / (n_ref - 1)
        integral = h * (integrand[0] + integrand[-1] + 
                       4*np.sum(integrand[1:-1:2]) + 
                       2*np.sum(integrand[2:-1:2]))
        return z * integral
    
    def compute_errors(nodes, weights, z_values, cosmologies):
        """Compute relative errors for all test points."""
        errors = []
        for w0, wa in cosmologies:
            for z in z_values:
                a_vals = 1.0 / (1.0 + z * nodes)
                E_vals = E_of_a(a_vals, w0, wa)
                quad_val = np.sum(weights / E_vals)
                ref_val = reference_integral(z, w0, wa)
                rel_err = np.abs(quad_val - ref_val) / ref_val
                errors.append(rel_err)
        return np.array(errors)
    
    def objective_chebyshev(params):
        """Minimize worst-case (max) relative error."""
        if time.time() > deadline:
            return 1e10
        
        # Unpack and transform
        nodes_raw = params[:8]
        weights_raw = params[8:16]
        
        # Nodes: cumulative sum of exp for strictly increasing in (0,1)
        t = np.exp(nodes_raw)
        cumsum_t = np.cumsum(t)
        nodes = cumsum_t / cumsum_t[-1]
        nodes = np.clip(nodes, 1e-8, 1.0 - 1e-8)
        
        # Weights: softmax for positive, sum to 1
        w_exp = np.exp(weights_raw - np.max(weights_raw))
        weights = w_exp / np.sum(w_exp)
        
        # Test points
        z_test = [0.1, 0.5, 1.0, 2.0, 5.0]
        
        errors = compute_errors(nodes, weights, z_test, cosmologies)
        return np.max(errors)
    
    def objective_smooth(params):
        """Smooth objective: sum of squared errors (easier to optimize)."""
        if time.time() > deadline:
            return 1e10
        
        nodes_raw = params[:8]
        weights_raw = params[8:16]
        
        t = np.exp(nodes_raw)
        cumsum_t = np.cumsum(t)
        nodes = cumsum_t / cumsum_t[-1]
        nodes = np.clip(nodes, 1e-8, 1.0 - 1e-8)
        
        w_exp = np.exp(weights_raw - np.max(weights_raw))
        weights = w_exp / np.sum(w_exp)
        
        z_test = [0.1, 0.5, 1.0, 2.0, 5.0]
        errors = compute_errors(nodes, weights, z_test, cosmologies)
        return np.sum(errors**2)
    
    # Initial guess: Gauss-Legendre mapped to [0,1]
    from scipy.special import roots_legendre
    gl_nodes, gl_weights = roots_legendre(8)
    gl_nodes_01 = 0.5 * (gl_nodes + 1)
    
    # Invert parameterization for nodes
    t_init = np.zeros(8)
    t_init[0] = gl_nodes_01[0]
    for i in range(1, 8):
        t_init[i] = gl_nodes_01[i] - gl_nodes_01[i-1]
    nodes_raw_init = np.log(t_init + 1e-10)
    
    # Invert for weights
    weights_raw_init = np.log(gl_weights + 1e-10) - np.log(gl_weights + 1e-10).max()
    
    params_init = np.concatenate([nodes_raw_init, weights_raw_init])
    
    # Phase 1: Optimize smooth objective with L-BFGS-B
    best_params = params_init
    best_score = objective_smooth(params_init)
    
    try:
        result1 = minimize(objective_smooth, params_init, method='L-BFGS-B',
                          options={'maxiter': 100, 'maxfun': 3000})
        if result1.fun < best_score:
            best_params = result1.x
            best_score = result1.fun
    except Exception:
        pass
    
    # Phase 2: Refine with Chebyshev (worst-case) objective
    if time.time() < deadline - 10:
        try:
            result2 = minimize(objective_chebyshev, best_params, method='Nelder-Mead',
                              options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            if result2.fun < objective_chebyshev(best_params):
                best_params = result2.x
        except Exception:
            pass
    
    # Extract final nodes and weights
    params_opt = best_params
    t = np.exp(params_opt[:8])
    cumsum_t = np.cumsum(t)
    nodes = cumsum_t / cumsum_t[-1]
    nodes = np.clip(nodes, 1e-8, 1.0 - 1e-8)
    
    w_exp = np.exp(params_opt[8:16] - np.max(params_opt[8:16]))
    weights = w_exp / np.sum(w_exp)
    
    # Final validation: ensure constraints
    weights = weights / np.sum(weights)
    
    if not np.all(np.diff(nodes) > 1e-10):
        # Fallback to Gauss-Legendre
        nodes = gl_nodes_01
        weights = gl_weights
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
