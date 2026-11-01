"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 115.0  # Safety margin before 120s limit

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

def _smooth_objective(x, T=5.0, eps=1e-8):
    """
    Smooth approximation of the objective for gradient-based optimization.
    x is the parameter vector, h = tanh(x) to keep values in (-1, 1).
    """
    h = np.tanh(x)
    conv = np.convolve(h, h, mode='full')
    abs_conv = np.sqrt(conv**2 + eps)
    
    # Soft max approximation
    max_abs = np.max(abs_conv)
    # Stable log-sum-exp
    shifted = (abs_conv - max_abs) / T
    log_sum_exp = np.log(np.sum(np.exp(shifted))) + max_abs / T
    log_max = T * log_sum_exp
    
    # Smooth L1 norm
    sum_abs_h = np.sum(np.sqrt(h**2 + eps))
    
    value = log_max / (sum_abs_h ** 2)
    return value

def _optimize_from_start(x0, max_iter=300, ftol=1e-10):
    """
    Run L-BFGS-B optimization from a given starting point.
    """
    def objective(x):
        return _smooth_objective(x)
    
    res = minimize(objective, x0, method='L-BFGS-B', 
                   options={'maxiter': max_iter, 'ftol': ftol, 'gtol': 1e-8})
    h_opt = np.tanh(res.x)
    return h_opt, res.fun

def _local_search_refine(h, max_steps=5000, start_time=None):
    """
    Greedy local search: flip individual steps if it reduces the objective.
    This is a structural change from pure gradient descent.
    """
    h = h.copy()
    current_val = _compute_value(h)
    
    # Precompute for faster delta evaluation
    # Flipping h[i] from v to -v changes conv by:
    # conv_new[k] = conv_old[k] - 2*v*(h * e_i)[k]
    # where (h * e_i)[k] = h[k-i] (with appropriate bounds)
    # Actually, let's just recompute for simplicity given N=400 is small
    
    no_improve_count = 0
    step = 0
    
    while step < max_steps:
        if start_time is not None:
            if time.time() - start_time > DEADLINE - 5.0:
                break
        
        # Try flipping a random index
        idx = np.random.randint(N)
        old_val = h[idx]
        h[idx] = -old_val
        
        new_val = _compute_value(h)
        
        if new_val < current_val:
            current_val = new_val
            no_improve_count = 0
        else:
            h[idx] = old_val
            no_improve_count += 1
        
        step += 1
        
        # If no improvement for a while, break
        if no_improve_count > 200:
            break
    
    return h, current_val

def _try_multiple_starts(start_time):
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
    
    # Start 5: Block pattern with different block size
    x0_5 = np.zeros(N)
    for i in range(N):
        if (i // 100) % 2 == 0:
            x0_5[i] = 5.0
        else:
            x0_5[i] = -5.0
    h5, val5 = _optimize_from_start(x0_5)
    raw_val5 = _compute_value(h5)
    if raw_val5 < best_value:
        best_value = raw_val5
        best_h = h5
    
    return best_h, best_value

def construct():
    """
    Construct the signed step function heights.
    Returns a sequence of 400 signed step heights.
    """
    start_time = time.time()
    
    # Try multiple starts to find a better solution
    h_opt, val_opt = _try_multiple_starts(start_time)
    
    # Local search refinement
    if h_opt is not None:
        h_opt, val_opt = _local_search_refine(h_opt, max_steps=3000, start_time=start_time)
    
    # If optimization failed or value is too high, fall back to seed
    if h_opt is None or val_opt > 0.0025:
        return np.ones(N)
    
    return h_opt
# EVOLVE-BLOCK-END
