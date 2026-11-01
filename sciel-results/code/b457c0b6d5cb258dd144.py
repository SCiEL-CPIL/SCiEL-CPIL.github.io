"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _c3_upper_bound(h):
    """
    Compute the upper bound on C3 for a signed step function with heights h.
    
    The metric is:
    upper_bound = max_k |convolve(h, h)[k]| / (sum_i |h_i|)^2
    
    We minimize this using gradient-based optimization.
    """
    def objective(x):
        h = np.tanh(x)  # Bounded in (-1, 1)
        conv = np.convolve(h, h, mode='full')
        # Use smooth absolute value and smooth max
        eps = 1e-8
        abs_conv = np.sqrt(conv**2 + eps)
        # Soft max with temperature
        T = 5.0
        # To avoid overflow, subtract max
        max_abs = np.max(abs_conv)
        log_sum_exp = np.log(np.sum(np.exp((abs_conv - max_abs) / T))) + max_abs / T
        log_max = T * log_sum_exp
        sum_abs_h = np.sum(np.sqrt(h**2 + eps))
        value = log_max / (sum_abs_h**2)
        return value
    
    # Try multiple random initializations to escape local minima
    best_h = None
    best_value = float('inf')
    
    # Start with the seed initialization (all positive)
    x0 = np.full(N, 5.0)
    res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12})
    h_opt = np.tanh(res.x)
    val = objective(res.x)
    if val < best_value:
        best_value = val
        best_h = h_opt
    
    # Try some random initializations
    np.random.seed(42)
    for trial in range(3):
        # Random signs with some structure
        x0 = np.random.randn(N) * 2.0
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12})
        h_opt = np.tanh(res.x)
        val = objective(res.x)
        if val < best_value:
            best_value = val
            best_h = h_opt
    
    return best_h


def construct():
    return _c3_upper_bound(np.full(N, 5.0))
# EVOLVE-BLOCK-END
