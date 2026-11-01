"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _compute_value(h):
    """
    Compute the raw upper bound value for a given height vector h.
    Metric: max_k |conv(h,h)_k| / (sum_i |h_i|)^2
    """
    conv = np.convolve(h, h, mode='full')
    max_abs_conv = np.max(np.abs(conv))
    sum_abs_h = np.sum(np.abs(h))
    if sum_abs_h == 0:
        return float('inf')
    return max_abs_conv / (sum_abs_h ** 2)


def _smooth_objective(x, T=10.0, eps=1e-8):
    """
    Smooth approximation of the objective for gradient-based optimization.
    x is the parameter vector, h = tanh(x) to keep values in (-1, 1).
    """
    h = np.tanh(x)
    conv = np.convolve(h, h, mode='full')
    abs_conv = np.sqrt(conv**2 + eps)
    
    # Soft max approximation
    max_abs = np.max(abs_conv)
    log_sum_exp = np.log(np.sum(np.exp((abs_conv - max_abs) / T))) + max_abs / T
    log_max = T * log_sum_exp
    
    # Smooth L1 norm
    sum_abs_h = np.sum(np.sqrt(h**2 + eps))
    
    value = log_max / (sum_abs_h ** 2)
    return value


def _optimize_from_start(x0, max_iter=500, ftol=1e-10):
    """
    Run L-BFGS-B optimization from a given starting point.
    """
    def objective(x):
        return _smooth_objective(x)
    
    res = minimize(objective, x0, method='L-BFGS-B', 
                   options={'maxiter': max_iter, 'ftol': ftol, 'gtol': 1e-8})
    h_opt = np.tanh(res.x)
    return h_opt, res.fun


def _try_multiple_starts():
    """
    Try multiple random starting points to escape local minima.
    """
    best_h = None
    best_value = float('inf')
    
    # Start 1: All ones (seed-like)
    x0_1 = np.full(N, 5.0)
    h1, val1 = _optimize_from_start(x0_1)
    raw_val1 = _compute_value(h1)
    if raw_val1 < best_value:
        best_value = raw_val1
        best_h = h1
    
    # Start 2: Alternating signs
    x0_2 = np.zeros(N)
    for i in range(N):
        x0_2[i] = 5.0 if i % 2 == 0 else -5.0
    h2, val2 = _optimize_from_start(x0_2)
    raw_val2 = _compute_value(h2)
    if raw_val2 < best_value:
        best_value = raw_val2
        best_h = h2
    
    # Start 3: Random
    np.random.seed(42)
    x0_3 = np.random.randn(N) * 2.0
    h3, val3 = _optimize_from_start(x0_3)
    raw_val3 = _compute_value(h3)
    if raw_val3 < best_value:
        best_value = raw_val3
        best_h = h3
    
    # Start 4: Block pattern
    x0_4 = np.zeros(N)
    for i in range(N):
        if (i // 50) % 2 == 0:
            x0_4[i] = 5.0
        else:
            x0_4[i] = -5.0
    h4, val4 = _optimize_from_start(x0_4)
    raw_val4 = _compute_value(h4)
    if raw_val4 < best_value:
        best_value = raw_val4
        best_h = h4
    
    return best_h, best_value


def construct():
    """
    Construct the signed step function heights.
    Returns a sequence of 400 signed step heights.
    """
    # Try multiple starts to find a better solution
    h_opt, val_opt = _try_multiple_starts()
    
    # If optimization failed or value is too high, fall back to seed
    if h_opt is None or val_opt > 0.0025:
        return np.ones(N)
    
    return h_opt
# EVOLVE-BLOCK-END
