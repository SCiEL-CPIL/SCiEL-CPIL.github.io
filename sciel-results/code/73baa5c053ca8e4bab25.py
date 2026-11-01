"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _c3_upper_bound(h):
    """
    Minimize the sidelobe peak of the autoconvolution relative to the square of the L1 norm.
    Metric: V(h) = max_{k != 0} |(h*h)_k| / (sum |h_i|)^2
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    def objective(x):
        if time.time() > deadline:
            return 1e9
        h = np.tanh(x)
        conv = np.convolve(h, h, mode='full')
        # Exclude lag 0 (index N-1 in 'full' mode for length N)
        # Indices 0 to N-2 and N to 2N-2 are sidelobes
        sidelobes = np.concatenate([conv[:N-1], conv[N:]])
        
        # Smooth max
        T = 5.0
        # Use log-sum-exp for stability
        max_val = np.max(sidelobes)
        log_sum_exp = max_val + T * np.log(np.sum(np.exp((sidelobes - max_val) / T)))
        soft_max = log_sum_exp / T
        
        sum_abs_h = np.sum(np.abs(h))
        if sum_abs_h < 1e-6:
            return 1e9
        value = soft_max / (sum_abs_h**2)
        return value

    # Initialize with random signs to avoid local minima of all-ones
    np.random.seed(42)
    x0 = np.random.uniform(-5, 5, N)
    
    # Try a few random restarts if time permits
    best_h = np.tanh(x0)
    best_val = objective(x0)
    
    for restart in range(5):
        if time.time() > deadline - 20:
            break
            
        if restart > 0:
            x0 = np.random.uniform(-5, 5, N)
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-9})
            if res.fun < best_val:
                best_val = res.fun
                best_h = np.tanh(res.x)
        except Exception:
            pass

    return best_h

def construct():
    return _c3_upper_bound(np.zeros(N))
# EVOLVE-BLOCK-END
