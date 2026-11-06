import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
import time

# Constants for cosmology
OM = 0.3
OR = 1e-4
ODE = 1.0 - OM - OR

def E_of_a(a, w0, w1):
    """
    Compute E(a) = H(a)/H0 for CPL dark energy model.
    """
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
    """
    Compute exact dimensionless luminosity distance.
    """
    if z <= 0:
        return 0.0
        
    def integrand(z_prime):
        a_prime = 1.0 / (1.0 + z_prime)
        return 1.0 / E_of_a(a_prime, w0, w1)
        
    try:
        val, err = quad(integrand, 0, z, limit=200)
        return (1.0 + z) * val
    except Exception:
        return 0.0

def quad_dL(z, nodes, weights, w0, w1):
    """
    Compute approximate dimensionless luminosity distance using the quadrature rule.
    """
    if z <= 0:
        return 0.0
        
    a_vals = 1.0 / (1.0 + z * nodes)
    e_vals = E_of_a(a_vals, w0, w1)
    sum_term = np.sum(weights / e_vals)
    return (1.0 + z) * z * sum_term

def fit_weights_for_nodes(nodes, test_data):
    """
    Given fixed nodes, find optimal weights via least-squares.
    For each (z, w0, w1), the quadrature computes:
      (1+z) * z * sum_i w_i / E(a_i)
    We want this to match exact_dL(z, w0, w1).
    
    So: sum_i w_i * [z / E(a_i)] = exact_dL(z, w0, w1) / (1+z)
    
    This is a linear system in w_i.
    """
    n_nodes = len(nodes)
    n_samples = len(test_data)
    
    # Build A matrix: A[k, i] = z_k / E(a_{k,i})
    # b vector: b[k] = exact_dL(z_k, w0_k, w1_k) / (1+z_k)
    A = np.zeros((n_samples, n_nodes))
    b = np.zeros(n_samples)
    
    for k, (z, w0, w1) in enumerate(test_data):
        a_vals = 1.0 / (1.0 + z * nodes)
        e_vals = E_of_a(a_vals, w0, w1)
        A[k, :] = z / e_vals
        exact = exact_dL(z, w0, w1)
        b[k] = exact / (1.0 + z)
    
    # Solve least squares with non-negativity constraint
    # Use scipy's nnls or manual projection
    from scipy.optimize import nnls
    w, rnorm = nnls(A, b)
    
    # Ensure weights sum to 1
    w_sum = np.sum(w)
    if w_sum > 1e-15:
        w = w / w_sum
    else:
        w = np.ones(n_nodes) / n_nodes
    
    return w

def compute_worst_error(nodes, weights, test_data):
    """Compute worst-case relative error."""
    max_err = 0.0
    for z, w0, w1 in test_data:
        exact = exact_dL(z, w0, w1)
        approx = quad_dL(z, nodes, weights, w0, w1)
        if exact > 1e-10:
            rel_err = abs((approx - exact) / exact)
            if rel_err > max_err:
                max_err = rel_err
    return max_err

def node_family_beta(alpha, beta, n=8):
    """Beta distribution quantiles as nodes."""
    from scipy.stats import beta as beta_dist
    u = (np.arange(n) + 0.5) / n
    return beta_dist.ppf(u, alpha, beta)

def node_family_power(p, n=8):
    """Power-law spaced nodes: x_i = ((i+0.5)/n)^p"""
    u = (np.arange(n) + 0.5) / n
    return u ** p

def node_family_log(n=8):
    """Log-spaced nodes in [0.01, 0.99]"""
    return np.geomspace(0.01, 0.99, n)

def node_family_uniform(n=8):
    """Uniform midpoint nodes."""
    return (np.arange(n) + 0.5) / n

def node_family_chebyshev(n=8):
    """Chebyshev nodes mapped to [0,1]"""
    k = np.arange(n)
    x = np.cos(np.pi * (2*k + 1) / (2*n))
    return (1.0 - x) / 2.0  # Map from [-1,1] to [0,1]

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin from 90s
    
    # Define test data for optimization
    zs = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0]
    w0s = [-1.5, -1.0, -0.5, 0.0, 0.5]
    w1s = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    
    test_data = []
    for z in zs:
        for w0 in w0s:
            for w1 in w1s:
                test_data.append((z, w0, w1))
    
    # Precompute exact values for speed
    exact_cache = {}
    for z, w0, w1 in test_data:
        key = (z, w0, w1)
        exact_cache[key] = exact_dL(z, w0, w1)
    
    def compute_worst_error_cached(nodes, weights):
        max_err = 0.0
        for z, w0, w1 in test_data:
            key = (z, w0, w1)
            exact = exact_cache[key]
            approx = quad_dL(z, nodes, weights, w0, w1)
            if exact > 1e-10:
                rel_err = abs((approx - exact) / exact)
                if rel_err > max_err:
                    max_err = rel_err
        return max_err
    
    # Strategy: Try multiple node families, fit weights via NNLS for each
    # Then do a local refinement of the best nodes
    
    best_nodes = node_family_uniform()
    best_weights = np.ones(8) / 8
    best_err = np.inf
    
    # Define node families to try
    node_candidates = []
    
    # Uniform
    node_candidates.append(('uniform', node_family_uniform()))
    
    # Chebyshev
    node_candidates.append(('chebyshev', node_family_chebyshev()))
    
    # Beta distribution with various parameters
    for alpha in [0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
        for beta_p in [0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
            if alpha > 0 and beta_p > 0:
                try:
                    nodes = node_family_beta(alpha, beta_p)
                    if nodes[0] > 1e-6 and nodes[-1] < 1 - 1e-6:
                        node_candidates.append((f'beta_{alpha}_{beta_p}', nodes))
                except:
                    pass
    
    # Power-law spaced
    for p in [0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]:
        nodes = node_family_power(p)
        if nodes[0] > 1e-6 and nodes[-1] < 1 - 1e-6:
            node_candidates.append((f'power_{p}', nodes))
    
    # Log-spaced
    node_candidates.append(('log', node_family_log()))
    
    # Evaluate each node family
    for name, nodes in node_candidates:
        if time.time() - start_time > deadline:
            break
        
        # Ensure nodes are in (0, 1) and strictly increasing
        nodes = np.clip(nodes, 1e-6, 1 - 1e-6)
        if not np.all(np.diff(nodes) > 1e-12):
            continue
        
        weights = fit_weights_for_nodes(nodes, test_data)
        err = compute_worst_error_cached(nodes, weights)
        
        if err < best_err:
            best_err = err
            best_nodes = nodes.copy()
            best_weights = weights.copy()
    
    # Local refinement of best nodes using gradient-free optimization
    # Only optimize nodes, re-fit weights at each step
    if time.time() - start_time < deadline - 10:
        def node_objective(node_params):
            # node_params: 8 values, use tanh to map to (0,1)
            nodes = 0.5 * (1.0 + np.tanh(node_params))
            nodes = np.sort(nodes)
            
            # Ensure strictly increasing with minimum spacing
            min_spacing = 1e-4
            for i in range(1, 8):
                if nodes[i] - nodes[i-1] < min_spacing:
                    nodes[i] = nodes[i-1] + min_spacing
            
            if nodes[0] < 1e-5 or nodes[-1] > 1 - 1e-5:
                return 1.0
            
            weights = fit_weights_for_nodes(nodes, test_data)
            return compute_worst_error_cached(nodes, weights)
        
        # Start from best nodes found
        init_params = np.arctanh(2.0 * best_nodes - 1.0)
        
        try:
            result = minimize(
                node_objective,
                init_params,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            refined_nodes = 0.5 * (1.0 + np.tanh(result.x))
            refined_nodes = np.sort(refined_nodes)
            
            # Ensure valid
            min_spacing = 1e-4
            for i in range(1, 8):
                if refined_nodes[i] - refined_nodes[i-1] < min_spacing:
                    refined_nodes[i] = refined_nodes[i-1] + min_spacing
            
            if refined_nodes[0] > 1e-5 and refined_nodes[-1] < 1 - 1e-5:
                refined_weights = fit_weights_for_nodes(refined_nodes, test_data)
                refined_err = compute_worst_error_cached(refined_nodes, refined_weights)
                
                if refined_err < best_err:
                    best_err = refined_err
                    best_nodes = refined_nodes.copy()
                    best_weights = refined_weights.copy()
        except Exception:
            pass
    
    # Final validation with a broader test set
    # Use the best rule found
    if time.time() - start_time < deadline - 5:
        # Verify nodes are strictly increasing in [0,1]
        if not np.all(np.diff(best_nodes) > 1e-12):
            best_nodes = node_family_uniform()
            best_weights = np.ones(8) / 8
        
        if best_nodes[0] <= 0 or best_nodes[-1] >= 1:
            best_nodes = node_family_uniform()
            best_weights = np.ones(8) / 8
        
        # Normalize weights
        best_weights = best_weights / np.sum(best_weights)
        
        # Ensure all weights positive
        if np.any(best_weights <= 0):
            best_weights = np.ones(8) / 8
    
    return {"nodes": best_nodes, "weights": best_weights}
