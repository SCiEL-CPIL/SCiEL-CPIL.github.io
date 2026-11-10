import numpy as np
from scipy.optimize import minimize

def _cpl_e(a, wa0, wa1):
    """
    Compute E(a) = H(a)/H0 for a CPL (Chesterman-Pearson-Linder) model.
    Omega_m = 0.3, Omega_L = 0.7.
    E^2(a) = Omega_m / a^3 + Omega_L * (a / (1 + a*wa1))^(3*(1+wa0))
    """
    omega_m = 0.3
    omega_l = 0.7
    # Handle a=0 case if it occurs, though nodes are in (0,1) so a in (1/(1+z), 1]
    # z is in [0, z_max], so a is in [1/(1+z_max), 1]
    # If z=0, a=1. If z is large, a is small.
    
    # Vectorized calculation
    term_m = omega_m / (a**3)
    base = a / (1.0 + a * wa1)
    # Ensure base is positive for fractional power
    # Since a > 0 and wa1 is typically not -1/a, base should be positive.
    # For robustness, clip base to a small positive number if it goes negative due to numerical issues
    base = np.maximum(base, 1e-15)
    exponent = 3.0 * (1.0 + wa0)
    term_l = omega_l * (base ** exponent)
    
    return np.sqrt(term_m + term_l)

def _compute_errors(params, z_samples, wa0_samples, wa1_samples, n_dense=1000):
    """
    Compute relative errors for a given set of nodes and weights.
    params: [node_params (8), weight_params (8)]
    node_params are used to generate nodes via cumsum(exp(params))
    weight_params are used to generate weights via exp(params)
    """
    node_params = params[:8]
    weight_params = params[8:16]
    
    # Generate nodes: strictly increasing in (0, 1)
    # d_i = exp(node_params[i]), nodes = cumsum(d) / sum(d)
    diffs = np.exp(node_params)
    nodes = np.cumsum(diffs)
    nodes = nodes / nodes[-1]
    
    # Generate weights: positive, normalized to sum to 1
    weights = np.exp(weight_params)
    weights = weights / np.sum(weights)
    
    errors = []
    
    for z in z_samples:
        for wa0 in wa0_samples:
            for wa1 in wa1_samples:
                # True integral: ∫_0^1 (1/E(a(z*u))) du
                # a(z*u) = 1 / (1 + z*u)
                
                # Dense integration for reference
                u_dense = np.linspace(0, 1, n_dense + 1)
                u_mid = 0.5 * (u_dense[:-1] + u_dense[1:])
                du = u_dense[1] - u_dense[0]
                
                a_mid = 1.0 / (1.0 + z * u_mid)
                E_mid = _cpl_e(a_mid, wa0, wa1)
                integrand = 1.0 / E_mid
                true_val = np.sum(integrand) * du
                
                # Quadrature approximation
                a_quad = 1.0 / (1.0 + z * nodes)
                E_quad = _cpl_e(a_quad, wa0, wa1)
                approx_val = np.sum(weights / E_quad)
                
                if true_val > 1e-12 and approx_val > 1e-12:
                    rel_err = abs(approx_val - true_val) / true_val
                    errors.append(rel_err)
    
    return np.array(errors), nodes, weights

def _objective_smooth(params, z_samples, wa0_samples, wa1_samples):
    """
    Smooth objective: sum of squared relative errors.
    """
    errors, _, _ = _compute_errors(params, z_samples, wa0_samples, wa1_samples)
    return np.sum(errors**2)

def _objective_worst(params, z_samples, wa0_samples, wa1_samples):
    """
    Worst-case objective: max relative error.
    """
    errors, _, _ = _compute_errors(params, z_samples, wa0_samples, wa1_samples)
    return np.max(errors)

def construct():
    # Define validation grid
    # z range: typical luminosity distance measurements go up to z~2 or 3
    z_samples = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0])
    
    # CPL parameters: typical ranges for dark energy
    # wa0 (w0) and wa1 (w1)
    wa0_samples = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2])
    wa1_samples = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    
    # Initial guess: Gauss-Legendre nodes and weights
    # Gauss-Legendre nodes and weights for n=8 on [0,1]
    x_gl, w_gl = np.polynomial.legendre.leggauss(8)
    # Map from [-1, 1] to [0, 1]
    nodes_init = 0.5 * (x_gl + 1.0)
    weights_init = 0.5 * w_gl
    
    # Invert to get params
    # nodes = cumsum(exp(node_params)) / sum(exp(node_params))
    # This is tricky to invert exactly. Instead, we can just start with random positive params
    # or use the midpoint rule as a start which is easier to parametrize.
    
    # Let's start with uniform nodes and weights
    nodes_start = (np.arange(8) + 0.5) / 8.0
    weights_start = np.full(8, 1.0/8.0)
    
    # We need params such that cumsum(exp(p)) / sum(exp(p)) = nodes_start
    # This is not trivial to invert. 
    # Alternative: Optimize nodes and weights directly with constraints?
    # No, the parameterization is better. 
    # Let's just use random positive values for the log-diffs.
    # If we want nodes ~ 1/8, 2/8... 
    # d_i = nodes_i - nodes_{i-1} (with nodes_0=0)
    # d_1 = 1/8, d_2 = 1/8...
    # So exp(node_params) ~ 1/8. node_params ~ log(1/8)
    
    node_params_init = np.full(8, np.log(1.0/8.0))
    weight_params_init = np.full(8, np.log(1.0/8.0))
    
    params_init = np.concatenate([node_params_init, weight_params_init])
    
    # Phase 1: L-BFGS-B on smooth objective
    # Use a subset of the grid for speed during optimization
    z_opt = z_samples[::2] # Every other z
    wa0_opt = wa0_samples[::2]
    wa1_opt = wa1_samples[::2]
    
    res1 = minimize(
        _objective_smooth,
        params_init,
        args=(z_opt, wa0_opt, wa1_opt),
        method='L-BFGS-B',
        options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10}
    )
    
    # Phase 2: Powell on worst-case objective with full grid
    res2 = minimize(
        _objective_worst,
        res1.x,
        args=(z_samples, wa0_samples, wa1_samples),
        method='Powell',
        options={'maxiter': 500, 'xtol': 1e-10, 'ftol': 1e-12}
    )
    
    # Choose the better result based on worst-case error
    err1, nodes1, weights1 = _compute_errors(res1.x, z_samples, wa0_samples, wa1_samples)
    err2, nodes2, weights2 = _compute_errors(res2.x, z_samples, wa0_samples, wa1_samples)
    
    if np.max(err2) < np.max(err1):
        nodes_opt = nodes2
        weights_opt = weights2
    else:
        nodes_opt = nodes1
        weights_opt = weights1
    
    # Post-processing: Ensure strict constraints
    # Nodes must be strictly increasing in [0, 1]
    # Weights positive and sum to 1
    
    # Check nodes
    if not np.all(np.diff(nodes_opt) > 0):
        # Fallback to midpoint if something went wrong
        nodes_opt = (np.arange(8) + 0.5) / 8.0
        weights_opt = np.full(8, 1.0/8.0)
        
    # Clip nodes to (0, 1) strictly
    # If any node is <= 0 or >= 1, adjust
    if nodes_opt[0] <= 0.0:
        nodes_opt[0] = 1e-10
    if nodes_opt[-1] >= 1.0:
        nodes_opt[-1] = 1.0 - 1e-10
        
    # Ensure strict increase
    for i in range(1, 8):
        if nodes_opt[i] <= nodes_opt[i-1]:
            nodes_opt[i] = nodes_opt[i-1] + 1e-10
            
    # Normalize weights
    weights_opt = weights_opt / np.sum(weights_opt)
    
    # Final check
    if np.sum(weights_opt) != 1.0:
        weights_opt = np.full(8, 1.0/8.0)
        
    return {"nodes": nodes_opt, "weights": weights_opt}
