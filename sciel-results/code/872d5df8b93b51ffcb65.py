"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import roots_legendre


def construct():
    """
    Construct an 8-node positive quadrature rule for dimensionless luminosity distance.
    
    We use a structured parameterization that guarantees strictly increasing nodes
    in (0,1) and positive weights summing to 1. We optimize a worst-case relative
    error objective across a representative set of CPL cosmologies and redshifts.
    """
    deadline = time.time() + 80  # 80s deadline, 10s safety margin
    
    # Representative CPL cosmologies: (w0, wa) pairs
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
        matter = Om_m / a**3
        if abs(w0 + wa) < 1e-10:
            de_term = Om_de * a**(-3*(1+w0))
        else:
            exponent = 3*wa/(1+w0+wa)
            de_term = Om_de * a**(-3*(1+w0+wa)) * (1 + wa*(1-a))**exponent
        return np.sqrt(matter + de_term)
    
    def reference_integral(z, w0, wa, n_ref=20001):
        """High-accuracy reference integral using many points with Simpson's rule."""
        u = np.linspace(0, 1, n_ref)
        a_vals = 1.0 / (1.0 + z * u)
        E_vals = E_of_a(a_vals, w0, wa)
        integrand = 1.0 / E_vals
        h = 1.0 / (n_ref - 1)
        integral = h * (integrand[0] + integrand[-1] + 4*np.sum(integrand[1:-1:2]) + 2*np.sum(integrand[2:-1:2]))
        return z * integral
    
    # Pre-compute reference integrals for all cosmologies and test z values
    z_test = [0.1, 0.5, 1.0, 2.0, 5.0]
    ref_cache = {}
    for w0, wa in cosmologies:
        for z in z_test:
            ref_cache[(w0, wa, z)] = reference_integral(z, w0, wa)
    
    def quadrature_value(nodes, weights, z, w0, wa):
        """Compute quadrature approximation of the integral."""
        a_vals = 1.0 / (1.0 + z * nodes)
        E_vals = E_of_a(a_vals, w0, wa)
        return np.sum(weights / E_vals)
    
    def objective(params):
        """Objective: maximize worst-case relative error (minimize negative)."""
        if time.time() > deadline:
            return 1e10
        
        # Unpack nodes and weights
        nodes_raw = params[:8]
        weights_raw = params[8:16]
        
        # Parameterize nodes using softplus to ensure strict increase in (0,1)
        # Let d_i = softplus(y_i) for i=0..6, and nodes[0] = softplus(y_7)
        # Then normalize so nodes are in (0,1)
        # Alternative: use cumulative sum of exponentials
        t = np.exp(nodes_raw)
        cumsum_t = np.cumsum(t)
        nodes = cumsum_t / cumsum_t[-1]
        nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
        
        # Weights: softmax to ensure positive and sum to 1
        w_exp = np.exp(weights_raw - np.max(weights_raw))
        weights = w_exp / np.sum(w_exp)
        
        # Compute worst-case relative error
        max_err = 0.0
        for w0, wa in cosmologies:
            for z in z_test:
                quad_val = quadrature_value(nodes, weights, z, w0, wa)
                ref_val = ref_cache[(w0, wa, z)]
                err = np.abs(quad_val - ref_val) / ref_val
                if err > max_err:
                    max_err = err
        
        return max_err
    
    # Initial guess: Gauss-Legendre nodes and weights mapped to [0,1]
    gl_nodes, gl_weights = roots_legendre(8)
    gl_nodes_01 = 0.5 * (gl_nodes + 1)
    
    # Invert the parameterization for initial guess
    t_init = np.zeros(8)
    t_init[0] = gl_nodes_01[0]
    for i in range(1, 8):
        t_init[i] = gl_nodes_01[i] - gl_nodes_01[i-1]
    nodes_raw_init = np.log(t_init + 1e-10)
    
    weights_raw_init = np.log(gl_weights + 1e-10) - np.log(gl_weights + 1e-10).max()
    
    params_init = np.concatenate([nodes_raw_init, weights_raw_init])
    
    # Optimize with L-BFGS-B
    result = None
    try:
        result = minimize(objective, params_init, method='L-BFGS-B', 
                          options={'maxiter': 300, 'maxfun': 10000, 'ftol': 1e-12, 'gtol': 1e-10})
        if time.time() > deadline:
            raise TimeoutError
    except Exception:
        result = None
    
    # Refine with Powell if time allows
    if result is not None and time.time() < deadline - 10:
        try:
            result2 = minimize(objective, result.x, method='Powell',
                              options={'maxiter': 200, 'maxfev': 10000})
            if result2.fun < result.fun:
                result = result2
        except Exception:
            pass
    
    if result is None:
        # Fallback to Gauss-Legendre
        nodes = gl_nodes_01
        weights = gl_weights
    else:
        params_opt = result.x
        t = np.exp(params_opt[:8])
        cumsum_t = np.cumsum(t)
        nodes = cumsum_t / cumsum_t[-1]
        nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
        
        w_exp = np.exp(params_opt[8:16] - np.max(params_opt[8:16]))
        weights = w_exp / np.sum(w_exp)
    
    # Ensure weights sum to 1 within tolerance
    weights = weights / np.sum(weights)
    
    # Ensure nodes are strictly increasing
    if not np.all(np.diff(nodes) > 0):
        nodes = gl_nodes_01
        weights = gl_weights
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
