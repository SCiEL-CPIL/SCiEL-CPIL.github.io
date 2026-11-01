"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _c3_upper_bound(h):
    """
    Compute the upper bound on C3 for a signed step function with heights h.
    
    The metric is: upper_bound = max_k |convolve(h, h)[k]| / (sum_i |h_i|)^2
    
    We minimize this by optimizing the continuous variable x where h = tanh(x).
    """
    def objective(x):
        h = np.tanh(x)
        conv = np.convolve(h, h, mode='full')
        eps = 1e-8
        abs_conv = np.sqrt(conv**2 + eps)
        
        # Soft max approximation for max(abs_conv)
        T = 5.0  # Temperature for soft max
        max_abs = np.max(abs_conv)
        # Use stable log-sum-exp
        shifted = (abs_conv - max_abs) / T
        log_sum_exp = np.log(np.sum(np.exp(shifted))) + max_abs / T
        log_max = T * log_sum_exp
        
        sum_abs_h = np.sum(np.sqrt(h**2 + eps))
        value = log_max / (sum_abs_h**2)
        return value
    
    # Start with a random initialization to escape local minima
    np.random.seed(42)
    x0 = np.random.normal(0, 1, N)
    
    # Use L-BFGS-B with multiple restarts to find a better minimum
    best_val = np.inf
    best_x = x0
    
    # Deadline guard: 145s out of 150s
    start_time = time.time()
    deadline = start_time + 145.0
    
    # First, try the "all ones" initialization which is known to be a local minimum
    x_ones = np.full(N, 5.0)
    res = minimize(objective, x_ones, method='L-BFGS-B', 
                  options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
    if res.fun < best_val:
        best_val = res.fun
        best_x = res.x
    
    # Check time
    if time.time() > deadline:
        return np.tanh(best_x)
    
    # Then do random restarts
    for i in range(10):
        if time.time() > deadline:
            break
        
        # Perturb the best solution found so far
        x0 = best_x + np.random.normal(0, 0.5, N)
        
        res = minimize(objective, x0, method='L-BFGS-B', 
                      options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
            
    # Final polish with tighter tolerance if time permits
    if time.time() < deadline - 10:
        res = minimize(objective, best_x, method='L-BFGS-B', 
                      options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x

    h_opt = np.tanh(best_x)
    return h_opt


def construct():
    return _c3_upper_bound(np.full(N, 5.0))
# EVOLVE-BLOCK-END
