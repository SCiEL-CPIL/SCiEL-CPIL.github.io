"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _compute_c3_bound(h):
    """
    Compute the actual C3 upper bound for a given height vector h.
    
    Metric: max_k |conv(h,h)[k]| / (sum_i |h_i|)^2
    
    This is the ratio of the peak autoconvolution magnitude to the squared L1 norm.
    """
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(np.abs(conv))
    l1_norm = np.sum(np.abs(h))
    if l1_norm == 0:
        return float('inf')
    return max_conv / (l1_norm ** 2)


def _smooth_objective(x, T=5.0, eps=1e-10):
    """
    Smooth approximation of the C3 bound for optimization.
    x is in the tanh parameter space.
    """
    h = np.tanh(x)
    conv = np.convolve(h, h, mode='full')
    abs_conv = np.sqrt(conv**2 + eps)
    
    # Soft max: T * log(sum(exp(abs_conv / T)))
    max_abs = np.max(abs_conv)
    log_sum_exp = np.log(np.sum(np.exp((abs_conv - max_abs) / T))) + max_abs / T
    smooth_max = T * log_sum_exp
    
    # Smooth L1 norm
    sum_abs_h = np.sum(np.sqrt(h**2 + eps))
    
    value = smooth_max / (sum_abs_h ** 2)
    return value


def _optimize_from_start(x0, T=5.0, maxiter=500):
    """Run L-BFGS-B optimization from a given starting point."""
    def objective(x):
        return _smooth_objective(x, T=T)
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', 
                      options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10})
        h_opt = np.tanh(res.x)
        actual_bound = _compute_c3_bound(h_opt)
        return h_opt, actual_bound
    except Exception as e:
        return None, float('inf')


def construct():
    """
    Construct a signed step function with 400 intervals that minimizes the C3 upper bound.
    
    Returns: array of 400 signed step heights
    """
    # Strategy: Try multiple random restarts to escape local minima
    # Start with the seed (all ones) and some random initializations
    
    best_h = None
    best_bound = float('inf')
    
    # Seed initialization: all ones
    x0_seed = np.full(N, 5.0)  # tanh(5) ≈ 0.9999
    h_seed, bound_seed = _optimize_from_start(x0_seed, T=5.0, maxiter=300)
    if h_seed is not None:
        best_h = h_seed
        best_bound = bound_seed
    
    # Try a few random restarts with mixed signs
    np.random.seed(42)
    for trial in range(3):
        # Random initialization in [-1, 1] via tanh
        x0_rand = np.random.randn(N) * 2.0
        h_rand, bound_rand = _optimize_from_start(x0_rand, T=5.0, maxiter=300)
        if h_rand is not None and bound_rand < best_bound:
            best_h = h_rand
            best_bound = bound_rand
    
    # If no valid solution found, return seed
    if best_h is None:
        return np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
