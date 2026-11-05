import numpy as np
from scipy.optimize import minimize

N = 400

def construct():
    # Start with a non-trivial initial guess to help optimization
    # Alternating signs with a small offset to avoid zero sum
    x0 = np.ones(N)
    x0[::2] = 1.0
    x0[1::2] = -0.9 # Sum is roughly 0.1 * N, non-zero
    
    # Soft max temperature
    T = 0.05
    
    def objective(h):
        s = np.sum(h)
        if abs(s) < 1e-6:
            return 1e10
        conv = np.convolve(h, h)
        # Use soft max for differentiability
        c_max = np.max(conv)
        # Numerical stability: subtract max before exp
        exp_conv = np.exp((conv - c_max) / T)
        soft_m = T * np.log(np.sum(exp_conv)) + c_max
        
        val = abs(2 * N * soft_m / (s * s))
        
        # Regularization to fix scale (scale invariance)
        # We want norm to be 1
        norm = np.linalg.norm(h)
        penalty = 0.01 * (norm - 1.0)**2
        
        return val + penalty

    # Bounds to prevent extreme values
    bounds = [(-5.0, 5.0)] * N
    
    # L-BFGS-B is suitable for large N
    try:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 2000, 'ftol': 1e-9, 'gtol': 1e-6})
        
        # Compare with seed
        def get_val(h):
            s = np.sum(h)
            if s == 0:
                return np.inf
            conv = np.convolve(h, h)
            m = np.max(conv)
            return abs(2 * N * m / (s * s))

        val_seed = get_val(np.ones(N))
        val_res = get_val(res.x)
        
        if val_res < val_seed:
            return res.x
        else:
            return np.ones(N)
            
    except Exception:
        return np.ones(N)
