"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 400


def construct():
    def calc_value(h):
        s = np.sum(h)
        if s == 0:
            return np.inf
        conv = np.convolve(h, h)
        return abs(2 * N * np.max(conv) / (s * s))

    best_h = np.ones(N)
    best_val = calc_value(best_h)
    
    rng = np.random.default_rng(123)
    
    # Try a few random starts with local optimization
    # Using Nelder-Mead to handle the non-smooth max and scale invariance
    for i in range(20):
        x0 = rng.normal(size=N)
        # Normalize to sum 1 to avoid singularity at sum=0
        x_sum = np.sum(x0)
        if x_sum == 0:
            continue
        x0 = x0 / x_sum
        
        def objective(x):
            s = np.sum(x)
            if abs(s) < 1e-10:
                return 1e10
            conv = np.convolve(x, x)
            return abs(2 * N * np.max(conv) / (s * s))
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12})
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_h = res.x
        except Exception:
            pass
        
    return best_h
