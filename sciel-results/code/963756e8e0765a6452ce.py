import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    timeout = 145.0 # Safety margin
    
    def objective(x):
        # x is the height vector
        # Normalize to L1 = 1
        x = np.abs(x)
        s = np.sum(x)
        if s == 0:
            return 1e9
        x = x / s
        
        # Calculate autoconvolution
        # c_k = sum x_i x_{k-i}
        # Use FFT for speed? N=50 is small, direct is fine.
        c = np.convolve(x, x, mode='full')
        
        # Supremum norm
        max_c = np.max(c)
        
        # L2 norm squared
        l2_sq = np.sum(x**2)
        
        if max_c == 0:
            return 0.0
            
        # We want to MINIMIZE max_c / l2_sq
        # This is equivalent to MAXIMIZING l2_sq / max_c
        return max_c / l2_sq

    def jac(x):
        # Analytic gradient is complex for max. Use numerical or SLSQP default.
        return None

    best_val = 1e9
    best_x = None
    
    # Try a few random starts
    np.random.seed(42)
    for i in range(10):
        if time.time() - start_time > timeout:
            break
            
        x0 = np.random.rand(N)
        
        res = minimize(objective, x0, method='SLSQP', jac=None, 
                       options={'maxiter': 100, 'ftol': 1e-10})
        
        if res.success:
            val = res.fun
            if val < best_val:
                best_val = val
                best_x = res.x
                
    if best_x is None:
        # Fallback to single impulse
        best_x = np.zeros(N)
        best_x[0] = 1.0
    else:
        # Normalize best_x
        best_x = np.abs(best_x)
        s = np.sum(best_x)
        if s > 0:
            best_x = best_x / s
        else:
            best_x = np.zeros(N)
            best_x[0] = 1.0
            
    return list(best_x)
