"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Construct an 8-node positive quadrature rule for the luminosity distance integral.
    Uses a Chebyshev spectral collocation approach combined with local refinement
    to minimize worst-case error across CPL dark energy models.
    """
    start_time = time.time()
    deadline = 50.0  # Safety margin before 60s limit
    
    # Cosmology constants
    OM = 0.3
    OR = 1e-4
    ODE = 1.0 - OM - OR
    
    def E_of_a(a, w0, w1):
        """Compute E(a) = H(a)/H0 for CPL dark energy model."""
        if a < 1e-10:
            return np.inf
        log_term = np.log(a)
        exp_arg = 3.0 * w1 * (a - 1.0 - log_term)
        if exp_arg > 700:
            exp_val = np.exp(700)
        else:
            exp_val = np.exp(exp_arg)
        power_term = a ** (-3.0 * (1.0 + w0))
        de_term = ODE * power_term * exp_val
        m_term = OM * a ** (-3.0)
        r_term = OR * a ** (-4.0)
        sum_term = m_term + r_term + de_term
        if sum_term < 0:
            return 0.0
        return np.sqrt(sum_term)
    
    def exact_dL(z, w0, w1):
        """Compute exact dimensionless luminosity distance using adaptive quadrature."""
        if z <= 0:
            return 0.0
        def integrand(z_prime):
            a_prime = 1.0 / (1.0 + z_prime)
            return 1.0 / E_of_a(a_prime, w0, w1)
        # Use Simpson's rule with fine grid for speed
        n_pts = 200
        z_grid = np.linspace(0, z, n_pts)
        a_grid = 1.0 / (1.0 + z_grid)
        e_vals = np.array([E_of_a(a, w0, w1) for a in a_grid])
        f_vals = 1.0 / e_vals
        # Simpson's rule
        h = z / (n_pts - 1)
        integral = h / 3 * (f_vals[0] + f_vals[-1] + 4 * np.sum(f_vals[1:-1:2]) + 2 * np.sum(f_vals[2:-1:2]))
        return (1.0 + z) * integral
    
    def quad_dL(z, nodes, weights, w0, w1):
        """Compute approximate luminosity distance using quadrature rule."""
        if z <= 0:
            return 0.0
        a_vals = 1.0 / (1.0 + z * nodes)
        e_vals = np.array([E_of_a(a, w0, w1) for a in a_vals])
        sum_term = np.sum(weights / e_vals)
        return (1.0 + z) * z * sum_term
    
    def chebyshev_nodes(n):
        """Generate Chebyshev nodes in [0, 1]."""
        k = np.arange(n)
        # Map from [-1, 1] to [0, 1]
        x = 0.5 * (1 - np.cos(np.pi * (k + 0.5) / n))
        return np.sort(x)
    
    def chebyshev_weights(n):
        """Generate Chebyshev weights in [0, 1] (normalized to sum to 1)."""
        k = np.arange(n)
        # Chebyshev weights in [-1, 1]
        w_cheb = np.pi / n
        # Adjust for endpoints if using clamped Chebyshev, but for collocation we use standard
        # For Gauss-Chebyshev quadrature, all weights are pi/n
        # But we want nodes strictly inside (0,1) and positive weights summing to 1
        # The standard Gauss-Chebyshev quadrature has nodes at cos((2k+1)pi/(2n))
        # and weights pi/n. Let's use that.
        x_cheb = np.cos(np.pi * (2 * k + 1) / (2 * n))  # in [-1, 1]
        x = 0.5 * (1 - x_cheb)  # map to [0, 1]
        x = np.sort(x)
        w = np.full(n, 1.0 / n)  # normalized weights
        return x, w
    
    # Define test cosmologies
    zs = [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    w0s = [-1.5, -1.2, -1.0, -0.8, -0.5, 0.0]
    w1s = [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    
    test_data = []
    for z in zs:
        for w0 in w0s:
            for w1 in w1s:
                test_data.append((z, w0, w1))
    
    def worst_case_error(nodes, weights):
        """Compute worst-case relative error."""
        max_err = 0.0
        for z, w0, w1 in test_data:
            exact = exact_dL(z, w0, w1)
            if exact == 0:
                continue
            approx = quad_dL(z, nodes, weights, w0, w1)
            rel_err = abs((approx - exact) / exact)
            if rel_err > max_err:
                max_err = rel_err
        return max_err
    
    # Start with Gauss-Chebyshev nodes and weights
    nodes, weights = chebyshev_weights(8)
    best_nodes = nodes.copy()
    best_weights = weights.copy()
    best_err = worst_case_error(nodes, weights)
    
    # Local refinement: perturb nodes and weights slightly to reduce error
    # Use a simple coordinate descent / Nelder-Mead style approach
    from scipy.optimize import minimize
    
    def objective(params):
        """Optimization objective: worst-case error."""
        v = params[:8]
        q = params[8:]
        
        # Sigmoid for nodes to ensure (0, 1)
        x = 1.0 / (1.0 + np.exp(-v))
        x = np.sort(x)
        
        # Softmax for weights
        q_max = np.max(q)
        exp_q = np.exp(q - q_max)
        w = exp_q / np.sum(exp_q)
        
        return worst_case_error(x, w)
    
    # Initial parameters from current best
    v_init = np.log(best_nodes / (1 - best_nodes))  # inverse sigmoid
    q_init = np.log(best_weights)  # inverse softmax (approximate)
    
    x0 = np.concatenate([v_init, q_init])
    bounds = [(-8.0, 8.0) for _ in range(16)]
    
    # Use Nelder-Mead for local refinement
    try:
        result = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
        
        if result.fun < best_err:
            best_params = result.x
            v_best = best_params[:8]
            q_best = best_params[8:]
            best_nodes = 1.0 / (1.0 + np.exp(-v_best))
            best_nodes = np.sort(best_nodes)
            q_max = np.max(q_best)
            exp_q = np.exp(q_best - q_max)
            best_weights = exp_q / np.sum(exp_q)
            best_err = result.fun
    except Exception:
        pass
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Fallback to current best
        pass
    
    # Final validation and fallback
    # Ensure nodes are strictly increasing in [0, 1]
    if not np.all(np.diff(best_nodes) > 1e-12) or best_nodes[0] <= 0 or best_nodes[-1] >= 1:
        # Fallback to midpoint rule
        n = 8
        best_nodes = (np.arange(n) + 0.5) / n
        best_weights = np.full(n, 1.0 / n)
    
    # Normalize weights
    best_weights = best_weights / np.sum(best_weights)
    
    # Final check: weights must be positive
    if np.any(best_weights <= 0):
        n = 8
        best_nodes = (np.arange(n) + 0.5) / n
        best_weights = np.full(n, 1.0 / n)
    
    return {"nodes": best_nodes, "weights": best_weights}
# EVOLVE-BLOCK-END
