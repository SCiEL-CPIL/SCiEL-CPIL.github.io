import numpy as np
from scipy.optimize import minimize
import time

N = 400

def autoconv_max_abs(a):
    """Compute max absolute value of autoconvolution of a."""
    n = len(a)
    # Full autoconvolution
    c = np.correlate(a, a, mode='full')
    # The 'full' mode gives length 2n-1, centered at n-1
    # We want the valid autoconvolution, which is the middle part
    # Actually for this problem, we probably want all lags
    return np.max(np.abs(c))

def objective(params):
    a = params
    # Normalize so sum = 1
    s = np.sum(a)
    if abs(s) < 1e-12:
        return 1e10
    a_norm = a / s
    return autoconv_max_abs(a_norm)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    best_a = np.ones(N)
    best_score = autoconv_max_abs(best_a) / (np.sum(best_a)**2)
    
    # Try random initializations with optimization
    np.random.seed(42)
    
    for trial in range(50):
        if time.time() > deadline - 30:
            break
            
        # Initialize with mostly positive values, some random
        a = np.random.randn(N)
        # Ensure non-zero sum
        a += np.random.randn() * 0.1
        
        res = minimize(objective, a, method='L-BFGS-B', 
                       options={'maxiter': 200, 'ftol': 1e-8})
        
        if time.time() > deadline - 30:
            break
            
        a_opt = res.x
        s = np.sum(a_opt)
        if abs(s) < 1e-12:
            continue
        a_norm = a_opt / s
        score = autoconv_max_abs(a_norm)
        
        if score < best_score:
            best_score = score
            best_a = a_norm.copy()
    
    return best_a
