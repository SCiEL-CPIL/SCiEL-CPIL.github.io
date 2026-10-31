import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    def ratio(h):
        h = np.maximum(h, 0)  # enforce nonnegativity
        l2_sq = np.sum(h**2)
        l1 = np.sum(h)
        if l1 < 1e-15:
            return 0.0
        # Compute autoconvolution (autocorrelation)
        c = np.correlate(h, h, mode='full')
        sup_c = np.max(np.abs(c))
        if sup_c < 1e-15:
            return 0.0
        return l2_sq / (l1 * sup_c)
    
    def neg_ratio(h):
        return -ratio(h)
    
    # Start with uniform
    best_h = np.ones(N)
    best_val = ratio(best_h)
    
    # Try several initializations
    inits = [
        np.ones(N),
        np.random.RandomState(42).rand(N),
        np.array([1.0 if i % 2 == 0 else 0.0 for i in range(N)]),
        np.array([1.0 if i % 2 == 1 else 0.0 for i in range(N)]),
        np.random.RandomState(123).rand(N),
        np.array([1.0] * 25 + [0.0] * 25),
        np.array([0.0] * 25 + [1.0] * 25),
    ]
    
    for init in inits:
        if time.time() > deadline:
            break
        try:
            res = minimize(neg_ratio, init, method='L-BFGS-B', 
                          bounds=[(0, None)] * N,
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < -best_val:
                best_val = -res.fun
                best_h = np.maximum(res.x, 0)
        except Exception:
            pass
    
    # Local polish with tighter tolerance
    if time.time() < deadline - 5:
        try:
            res = minimize(neg_ratio, best_h, method='L-BFGS-B',
                          bounds=[(0, None)] * N,
                          options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < -best_val:
                best_val = -res.fun
                best_h = np.maximum(res.x, 0)
        except Exception:
            pass
    
    # Ensure nonnegative
    best_h = np.maximum(best_h, 0)
    return list(best_h)
# EVOLVE-BLOCK-END
