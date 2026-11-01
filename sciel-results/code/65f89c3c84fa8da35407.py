"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _true_value(h):
    """Compute the exact metric: max |conv(h,h)| / (sum |h|)^2"""
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(np.abs(conv))
    sum_abs = np.sum(np.abs(h))
    if sum_abs == 0:
        return np.inf
    return max_conv / (sum_abs ** 2)


def _smooth_objective(x):
    """
    Smooth approximation of the metric for gradient-based optimization.
    x is the unconstrained parameter vector, h = tanh(x).
    """
    h = np.tanh(x)
    conv = np.convolve(h, h, mode='full')
    eps = 1e-10
    
    # Smooth absolute value for convolution terms
    abs_conv = np.sqrt(conv**2 + eps)
    
    # Soft max for the numerator
    # T=1.0 is a reasonable balance; smaller T is sharper but noisier gradients
    T = 1.0
    max_abs = np.max(abs_conv)
    # Numerical stability: subtract max
    log_sum_exp = np.log(np.sum(np.exp((abs_conv - max_abs) / T))) + max_abs / T
    soft_max = T * log_sum_exp
    
    # Smooth L1 norm for the denominator
    # Using sqrt(h^2 + eps) approximates |h|
    sum_abs_h = np.sum(np.sqrt(h**2 + eps))
    
    value = soft_max / (sum_abs_h ** 2)
    return value


def _optimize_from_start(x0):
    """Run L-BFGS-B from a given starting point x0."""
    res = minimize(_smooth_objective, x0, method='L-BFGS-B', 
                   options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-8})
    h_opt = np.tanh(res.x)
    return h_opt


def construct():
    """
    Multi-start optimization to find a signed step function minimizing the C3 upper bound.
    """
    best_h = None
    best_val = np.inf
    
    # 1. Start with the seed configuration (all positive)
    x0_seed = np.full(N, 5.0)
    h_seed = _optimize_from_start(x0_seed)
    val_seed = _true_value(h_seed)
    if val_seed < best_val:
        best_val = val_seed
        best_h = h_seed
    
    # 2. Try several random binary-like starts
    # Random signs mapped to tanh parameters: sign * 5.0
    np.random.seed(42) # Fixed seed for reproducibility in this step, but random enough
    for i in range(10):
        # Random signs
        signs = np.random.choice([-1, 1], size=N)
        x0_rand = signs * 5.0
        
        h_rand = _optimize_from_start(x0_rand)
        val_rand = _true_value(h_rand)
        
        if val_rand < best_val:
            best_val = val_rand
            best_h = h_rand
            
    return best_h
# EVOLVE-BLOCK-END
