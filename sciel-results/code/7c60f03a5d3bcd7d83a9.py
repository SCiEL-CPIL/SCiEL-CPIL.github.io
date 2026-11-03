import numpy as np
import time
import scipy.optimize

N = 400

def _autoconv_lin(x):
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    fx = np.fft.rfft(x, size)
    return np.fft.irfft(fx * fx, size)[:2 * n - 1]

def _metric_offdiag(x):
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv_lin(x)
    if len(c) > 1:
        return float(np.max(np.abs(c[1:]))) / (s * s)
    return 0.0

def _objective_soft_max(x, beta=50.0):
    """
    Soft max objective for off-diagonal metric.
    """
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e10
    c = _autoconv_lin(x)
    if len(c) <= 1:
        return 0.0
    abs_c = np.abs(c[1:])
    m = np.max(abs_c)
    if m < 1e-15:
        return 0.0
    soft_max = m + np.log(np.sum(np.exp(beta * (abs_c - m)))) / beta
    return soft_max / (s * s)

def construct():
    deadline = time.time() + 135.0
    rng = np.random.default_rng(42)
    
    # Baseline
    x_const = np.ones(N)
    best_x = x_const
    best_val = _metric_offdiag(x_const)
    
    # Initial candidates
    candidates = [
        x_const.copy(),
        np.hanning(N), # Positive, sum > 0
        rng.choice([-1, 1], size=N), # Random binary
        rng.normal(0, 1, N) # Random continuous
    ]
    
    # Evaluate initial candidates
    for x_c in candidates:
        val = _metric_offdiag(x_c)
        if val < best_val:
            best_x = x_c
            best_val = val
            
    # Optimization loop
    while time.time() < deadline:
        # Pick a random starting point from candidates or best_x
        if rng.random() < 0.5:
            x_init = best_x + rng.normal(0, 0.1, N)
        else:
            x_init = candidates[rng.integers(len(candidates))] + rng.normal(0, 0.1, N)
            
        try:
            res = scipy.optimize.minimize(
                _objective_soft_max, 
                x_init, 
                method='L-BFGS-B', 
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-7}
            )
            x_opt = res.x
            val_opt = _metric_offdiag(x_opt)
            
            if val_opt < best_val:
                best_x = x_opt
                best_val = val_opt
        except Exception:
            pass

    return best_x
