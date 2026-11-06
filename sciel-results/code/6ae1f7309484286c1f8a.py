import numpy as np
from scipy.optimize import differential_evolution
from scipy.integrate import quad
import time

# Constants for cosmology
OM = 0.3
OR = 1e-4
ODE = 1.0 - OM - OR

def E_of_a(a, w0, w1):
    """
    Compute E(a) = H(a)/H0 for CPL dark energy model.
    a: scale factor
    w0, w1: CPL parameters
    """
    # Handle a=0 or very small
    if a < 1e-10:
        return np.inf
    
    # Exponent part for DE
    # exp(3 * w1 * (a - 1 - ln(a)))
    # Power part: a^(-3 * (1 + w0))
    
    # Numerical stability for ln(a) when a is close to 1
    # Use np.log
    log_term = np.log(a)
    exp_arg = 3.0 * w1 * (a - 1.0 - log_term)
    
    # Avoid overflow in exp
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
        return 0.0 # Should not happen for physical models
        
    return np.sqrt(sum_term)

def exact_dL(z, w0, w1):
    """
    Compute exact dimensionless luminosity distance (without c/H0 factor, but including the integral part).
    The formula is dL/(c/H0) = (1+z) * Integral_0^z dz'/E(a')
    """
    if z <= 0:
        return 0.0
        
    def integrand(z_prime):
        a_prime = 1.0 / (1.0 + z_prime)
        return 1.0 / E_of_a(a_prime, w0, w1)
        
    try:
        val, err = quad(integrand, 0, z, limit=100)
        return (1.0 + z) * val
    except Exception:
        return 0.0

def quad_dL(z, nodes, weights, w0, w1):
    """
    Compute approximate dimensionless luminosity distance using the quadrature rule.
    Formula: (1+z) * z * sum_i weights[i] / E(a_i)
    where a_i = 1 / (1 + z * nodes[i])
    """
    if z <= 0:
        return 0.0
        
    a_vals = 1.0 / (1.0 + z * nodes)
    e_vals = E_of_a(a_vals, w0, w1)
    
    # Handle potential zeros or infs in E
    # If E is 0, 1/E is inf. 
    # For z>0, a<1, E should be finite and positive.
    
    sum_term = np.sum(weights / e_vals)
    return (1.0 + z) * z * sum_term

def get_nodes_weights(params):
    """
    Convert optimization parameters to nodes and weights.
    params: 16 values. First 8 for nodes (via sigmoid), last 8 for weights (via softmax).
    """
    v = params[:8]
    q = params[8:]
    
    # Sigmoid for nodes
    x = 1.0 / (1.0 + np.exp(-v))
    
    # Sort nodes to ensure strictly increasing
    x = np.sort(x)
    
    # Softmax for weights
    # Subtract max for stability
    q_max = np.max(q)
    exp_q = np.exp(q - q_max)
    w = exp_q / np.sum(exp_q)
    
    return x, w

def objective(params, test_data):
    """
    Compute worst-case relative error.
    """
    x, w = get_nodes_weights(params)
    
    max_err = 0.0
    
    for z, w0, w1 in test_data:
        exact = exact_dL(z, w0, w1)
        approx = quad_dL(z, x, w, w0, w1)
        
        if exact == 0:
            continue
            
        rel_err = abs((approx - exact) / exact)
        
        if rel_err > max_err:
            max_err = rel_err
            
    return max_err

def construct():
    start_time = time.time()
    deadline = 60.0 # Keep margin from 90s
    
    # Define test data: (z, w0, w1)
    # We want to cover a good range of behavior.
    # w0, w1 from reasonable CPL ranges
    # z from low to high
    
    zs = [0.1, 0.5, 1.0, 2.0, 4.0]
    w0s = [-1.5, -1.0, -0.5, 0.0]
    w1s = [-1.0, 0.0, 1.0, 2.0]
    
    test_data = []
    for z in zs:
        for w0 in w0s:
            for w1 in w1s:
                test_data.append((z, w0, w1))
                
    # To speed up, we can subsample the test data during optimization,
    # but use the full set for final verification if time permits.
    # For now, use all 80 points.
    
    # Initial bounds for params
    # v for nodes: sigmoid maps (-inf, inf) to (0, 1).
    # We want nodes spread in [0, 1]. 
    # v in [-5, 5] gives sigmoid in [0.006, 0.994].
    # q for weights: softmax. Any real numbers.
    # Bounds for q: [-5, 5] is reasonable.
    
    bounds = [(-5.0, 5.0) for _ in range(16)]
    
    # Use differential evolution
    # popsize=15, maxiter=50
    # tol=1e-6
    
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            args=(test_data,),
            maxiter=30, 
            popsize=15,
            tol=1e-6,
            seed=42,
            polish=True
        )
        
        best_params = result.x
        best_x, best_w = get_nodes_weights(best_params)
        
        # Verify time
        elapsed = time.time() - start_time
        if elapsed > deadline:
            # Fallback to midpoint rule if too slow (should not happen)
            n = 8
            return {"nodes": (np.arange(n) + 0.5) / n, "weights": np.full(n, 1.0 / n)}
            
        # Final check: ensure nodes are strictly increasing in [0, 1]
        # and weights sum to 1.
        # The sigmoid/softmax ensures this, but let's be safe.
        # If two nodes are identical due to sorting of close sigmoids, 
        # the "strictly increasing" might fail if they are exactly equal.
        # We can add a tiny jitter if needed, but sigmoid of distinct v's should be distinct.
        
        # Check strictly increasing
        if not np.all(np.diff(best_x) > 1e-12):
            # Fallback
            n = 8
            return {"nodes": (np.arange(n) + 0.5) / n, "weights": np.full(n, 1.0 / n)}
            
        # Check bounds
        if best_x[0] < 0 or best_x[-1] > 1:
             # Fallback
            n = 8
            return {"nodes": (np.arange(n) + 0.5) / n, "weights": np.full(n, 1.0 / n)}
            
        # Normalize weights just in case of floating point drift
        best_w = best_w / np.sum(best_w)
        
        return {"nodes": best_x, "weights": best_w}
        
    except Exception as e:
        # Fallback to midpoint rule on error
        n = 8
        return {"nodes": (np.arange(n) + 0.5) / n, "weights": np.full(n, 1.0 / n)}
