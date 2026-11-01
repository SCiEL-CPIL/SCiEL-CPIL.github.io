"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _raw_value(h):
    """
    Compute the actual raw value (upper bound on C3) for a given height vector h.
    Metric: max_k |conv(h, h)[k]| / (sum_i |h_i|)^2
    """
    conv = np.convolve(h, h, mode='full')
    max_abs_conv = np.max(np.abs(conv))
    sum_abs_h = np.sum(np.abs(h))
    if sum_abs_h == 0:
        return np.inf
    return max_abs_conv / (sum_abs_h ** 2)


def _smooth_objective(x, T=10.0):
    """
    Smooth approximation of the objective for gradient-based optimization.
    x is the parameter vector, h = tanh(x) in (-1, 1).
    """
    h = np.tanh(x)
    conv = np.convolve(h, h, mode='full')
    
    # Smooth absolute value
    eps = 1e-10
    abs_conv = np.sqrt(conv**2 + eps)
    
    # Smooth max using log-sum-exp
    max_abs = np.max(abs_conv)
    log_sum_exp = np.log(np.sum(np.exp((abs_conv - max_abs) / T))) + max_abs / T
    smooth_max = T * log_sum_exp
    
    # Smooth sum of absolute values
    sum_abs_h = np.sum(np.sqrt(h**2 + eps))
    
    value = smooth_max / (sum_abs_h ** 2)
    return value


def construct():
    # Start with the seed configuration: all ones
    # In tanh parameter space, tanh(5) ≈ 0.99991
    x0 = np.full(N, 5.0)
    
    # Try multiple initial conditions to escape local minima
    best_h = np.ones(N)
    best_val = _raw_value(best_h)
    
    # Primary optimization from seed
    res = minimize(
        _smooth_objective, 
        x0, 
        method='L-BFGS-B', 
        options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    h_opt = np.tanh(res.x)
    val_opt = _raw_value(h_opt)
    
    if val_opt < best_val:
        best_h = h_opt
        best_val = val_opt
    
    # Try a few different initializations to explore the landscape
    # 1. Random initialization
    np.random.seed(42)
    for trial in range(3):
        x_rand = np.random.randn(N) * 2.0
        res_rand = minimize(
            _smooth_objective,
            x_rand,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        h_rand = np.tanh(res_rand.x)
        val_rand = _raw_value(h_rand)
        
        if val_rand < best_val:
            best_h = h_rand
            best_val = val_rand
    
    # 2. Structured initialization: alternating signs
    x_alt = np.array([5.0 if i % 2 == 0 else -5.0 for i in range(N)])
    res_alt = minimize(
        _smooth_objective,
        x_alt,
        method='L-BFGS-B',
        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    h_alt = np.tanh(res_alt.x)
    val_alt = _raw_value(h_alt)
    
    if val_alt < best_val:
        best_h = h_alt
        best_val = val_alt
    
    # 3. Block initialization: half positive, half negative
    x_block = np.concatenate([np.full(N//2, 5.0), np.full(N//2, -5.0)])
    res_block = minimize(
        _smooth_objective,
        x_block,
        method='L-BFGS-B',
        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    h_block = np.tanh(res_block.x)
    val_block = _raw_value(h_block)
    
    if val_block < best_val:
        best_h = h_block
        best_val = val_block
    
    # Refine the best solution with a final optimization using lower temperature
    x_best = np.arctanh(np.clip(best_h, -0.9999, 0.9999))
    
    def _refine_objective(x):
        return _smooth_objective(x, T=20.0)
    
    res_refine = minimize(
        _refine_objective,
        x_best,
        method='L-BFGS-B',
        options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10}
    )
    h_final = np.tanh(res_refine.x)
    val_final = _raw_value(h_final)
    
    if val_final < best_val:
        best_h = h_final
        best_val = val_final
    
    return best_h
# EVOLVE-BLOCK-END
