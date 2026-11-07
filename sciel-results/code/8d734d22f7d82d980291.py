"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START
def compute_value(h):
    """Compute the value metric for given heights h."""
    h = np.asarray(h, dtype=np.float64)
    g = np.convolve(h, h, mode='full')
    m = len(g)
    # y = [0, g[0], ..., g[m-1], 0]
    y = np.zeros(m + 2)
    y[1:m+1] = g
    
    # spacing w: points are np.linspace(-0.5, 0.5, m+2)
    w = 1.0 / (m + 1)
    
    # l2 = sum over i in 0..m of (w/3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    y_i = y[:m+1]
    y_ip1 = y[1:m+2]
    l2 = (w / 3.0) * np.sum(y_i**2 + y_i * y_ip1 + y_ip1**2)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
    
    value = l2 / (norm_1 * norm_inf)
    return value


def negative_value(h):
    """Negative of the value for minimization."""
    return -compute_value(h)


def construct():
    # Deadline guard: 150s wall-clock limit, use 140s for safety
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Seed: 50 equal heights
    best_h = np.ones(N)
    best_val = compute_value(best_h)
    
    # Use Nelder-Mead optimization to find a flatter-topped autocorrelation
    # Start from the seed
    x0 = np.ones(N)
    
    # Define bounds: heights must be nonnegative
    # We'll use a transformation or constraints
    
    # Strategy: optimize with Nelder-Mead, using a smooth surrogate if needed
    # The objective is ratio l2/(norm_1 * norm_inf), which is smooth for positive g
    
    # First, try a direct Nelder-Mead optimization
    options = {
        'maxiter': 5000,
        'maxfev': 10000,
        'xatol': 1e-8,
        'fatol': 1e-10,
        'adaptive': True,
    }
    
    result = minimize(negative_value, x0, method='Nelder-Mead', options=options)
    
    # Check if we improved
    if result.fun < -best_val:
        best_h = np.maximum(result.x, 0)  # Ensure nonnegativity
        best_val = compute_value(best_h)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        return list(best_h)
    
    # Try additional restarts with different initial points
    # Use symmetric profiles that might give flatter autocorrelation
    
    # Try a plateau-like profile
    plateau_widths = [10, 15, 20, 25, 30]
    for pw in plateau_widths:
        if time.time() > deadline - 10:
            break
        h_init = np.zeros(N)
        center = N // 2
        half = pw // 2
        h_init[center - half:center + half + 1] = 1.0
        # Smooth edges
        if half > 1:
            h_init[center - half] = 0.5
            h_init[center + half] = 0.5
        
        res = minimize(negative_value, h_init, method='Nelder-Mead', options=options)
        if res.fun < -best_val:
            candidate = np.maximum(res.x, 0)
            val = compute_value(candidate)
            if val > best_val:
                best_val = val
                best_h = candidate
    
    # Try a Gaussian-like profile
    if time.time() < deadline - 10:
        for sigma in [3, 5, 7, 10]:
            if time.time() > deadline - 10:
                break
            x = np.arange(N)
            center = N / 2.0
            h_init = np.exp(-0.5 * ((x - center) / sigma) ** 2)
            
            res = minimize(negative_value, h_init, method='Nelder-Mead', options=options)
            if res.fun < -best_val:
                candidate = np.maximum(res.x, 0)
                val = compute_value(candidate)
                if val > best_val:
                    best_val = val
                    best_h = candidate
    
    # Final polish: high-precision local optimization around the best found
    if time.time() < deadline - 5:
        options_fine = {
            'maxiter': 3000,
            'maxfev': 6000,
            'xatol': 1e-10,
            'fatol': 1e-12,
            'adaptive': True,
        }
        res_fine = minimize(negative_value, best_h, method='Nelder-Mead', options=options_fine)
        if res_fine.fun < -best_val:
            candidate = np.maximum(res_fine.x, 0)
            val = compute_value(candidate)
            if val > best_val:
                best_val = val
                best_h = candidate
    
    return list(best_h)
# EVOLVE-BLOCK-END
