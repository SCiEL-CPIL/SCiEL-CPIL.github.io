import numpy as np
from scipy.optimize import minimize
import time

N = 600
START_TIME = time.time()
TIME_LIMIT = 150.0

def construct():
    # Initial guess: Box function
    x0 = np.ones(N)
    
    def objective(x):
        x = np.maximum(x, 0)
        conv = np.convolve(x, x, mode='full')
        max_conv = np.max(conv)
        sum_x = np.sum(x)
        if sum_x == 0:
            return 1e10
        return max_conv / (sum_x * sum_x)
    
    bounds = [(0, None)] * N
    
    try:
        # Phase 1: Coarse optimization
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 200, 'ftol': 1e-8})
        best_x = res.x if res.success else x0
        
        # Phase 2: Fine optimization
        if time.time() - START_TIME < TIME_LIMIT - 20:
            res2 = minimize(objective, best_x, method='L-BFGS-B', bounds=bounds, 
                            options={'maxiter': 100, 'ftol': 1e-12})
            if res2.success:
                best_x = res2.x
                
        return [float(v) for v in best_x]
        
    except Exception:
        return [1.0] * N
