# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

N = 400

def _objective(x):
    """
    Objective function: max_k |conv(h, h)[k]| / (sum_i |h_i|)^2
    h = tanh(x)
    """
    h = np.tanh(x)
    
    # Compute autoconvolution
    conv = np.convolve(h, h, mode='full')
    
    # Smooth absolute value and max
    eps = 1e-10
    abs_conv = np.sqrt(conv**2 + eps)
    
    # Soft max approximation
    T = 1.0
    max_abs = np.max(abs_conv)
    log_sum_exp = np.log(np.sum(np.exp((abs_conv - max_abs) / T))) + max_abs / T
    log_max = T * log_sum_exp
    
    # Denominator: (sum |h|)^2
    sum_abs_h = np.sum(np.sqrt(h**2 + eps))
    denom = sum_abs_h * sum_abs_h
    
    if denom < 1e-6:
        return 1e6
        
    return log_max / denom

def _c3_upper_bound(x0):
    """
    Wrapper to maintain signature similarity if needed, but construct calls _optimize.
    Actually, the seed had _c3_upper_bound(h) which did the optimization.
    I will make _c3_upper_bound do the optimization given an initial guess or default.
    """
    # We will use multiple random restarts inside the optimization logic
    # to find a good global minimum.
    
    best_h = None
    best_val = float('inf')
    
    rng = np.random.default_rng(42)
    
    # Number of restarts
    n_restarts = 10
    
    for _ in range(n_restarts):
        # Initialize with random signs, scaled to be near +/- 1
        # tanh(5) is very close to 1
        x_init = rng.choice([-1, 1], size=N) * 5.0
        
        # Optimization
        try:
            res = minimize(_objective, x_init, method='L-BFGS-B', 
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
            
            h_opt = np.tanh(res.x)
            val = _objective(res.x)
            
            if val < best_val:
                best_val = val
                best_h = h_opt
        except Exception:
            continue
            
    if best_h is None:
        # Fallback to all ones if optimization fails completely
        return np.ones(N)
        
    return best_h

def construct():
    # The seed program called _c3_upper_bound(np.full(N, 5.0))
    # I will call _c3_upper_bound with a dummy arg or just use the internal logic.
    # To keep the interface "construct() -> sequence", I just call the optimizer.
    return _c3_upper_bound(None)
# EVOLVE-BLOCK-END
