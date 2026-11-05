"""Optimized program for autocorrelation inequality task."""
import numpy as np
from scipy.optimize import minimize
import time

N = 400

def _compute_value(h):
    """Compute the exact evaluation value."""
    s = np.sum(h)
    if s == 0:
        return float('inf')
    conv = np.convolve(h, h)
    return abs(2 * N * np.max(conv) / (s ** 2))

def _smooth_max_surrogate(conv, beta=50.0):
    """Smooth approximation of max using log-sum-exp."""
    # Use log-sum-exp for numerical stability
    m = np.max(conv)
    return m + np.log(np.sum(np.exp(conv - m)) / len(conv)) / beta

def _objective(params, beta=50.0):
    """Objective: smooth max of autocorrelation, normalized."""
    # Reconstruct h from Fourier coefficients
    # params: [a0, a1, b1, a2, b2, ..., aK, bK]
    K = (len(params) - 1) // 2
    h = np.zeros(N)
    h += params[0] / 2
    for k in range(1, K + 1):
        h += params[1 + 2*(k-1)] * np.cos(2 * np.pi * k * np.arange(N) / N)
        h += params[2 + 2*(k-1)] * np.sin(2 * np.pi * k * np.arange(N) / N)
    
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h)
    sm = _smooth_max_surrogate(conv, beta)
    return sm / (s ** 2)

def construct():
    """Construct step heights minimizing autocorrelation peak."""
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    # Use a moderate number of Fourier modes
    K = 10  # 10 cosine + 10 sine + 1 DC = 21 parameters
    
    best_h = np.ones(N)
    best_val = _compute_value(best_h)
    
    # Multi-start optimization
    np.random.seed(42)
    n_starts = 20
    
    for i in range(n_starts):
        if time.time() > deadline:
            break
        
        # Initialize with random Fourier coefficients
        if i == 0:
            # Start near all-ones (seed)
            params0 = np.zeros(1 + 2*K)
            params0[0] = N  # DC term to make sum = N
        else:
            # Random initialization with some structure
            params0 = np.random.randn(1 + 2*K) * 0.1
            params0[0] = N * np.random.uniform(0.5, 1.5)
        
        # Optimize
        try:
            result = minimize(
                _objective,
                params0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10},
                args=(50.0,)
            )
            
            if result.success or result.fun < best_val:
                # Reconstruct h
                h = np.zeros(N)
                h += result.x[0] / 2
                for k in range(1, K + 1):
                    h += result.x[1 + 2*(k-1)] * np.cos(2 * np.pi * k * np.arange(N) / N)
                    h += result.x[2 + 2*(k-1)] * np.sin(2 * np.pi * k * np.arange(N) / N)
                
                val = _compute_value(h)
                if val < best_val:
                    best_val = val
                    best_h = h.copy()
                    
        except Exception:
            continue
    
    # Final polish: try to improve with higher beta (sharper max approximation)
    if time.time() < deadline - 20 and best_val < 2.0:
        # Re-optimize with higher beta for sharper max
        h = best_h
        # Extract current Fourier coefficients
        params_current = np.zeros(1 + 2*K)
        params_current[0] = 2 * np.sum(h) / N
        for k in range(1, K + 1):
            t = np.arange(N)
            params_current[1 + 2*(k-1)] = 2 * np.sum(h * np.cos(2 * np.pi * k * t / N)) / N
            params_current[2 + 2*(k-1)] = 2 * np.sum(h * np.sin(2 * np.pi * k * t / N)) / N
        
        try:
            result = minimize(
                _objective,
                params_current,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12},
                args=(200.0,)  # Higher beta for sharper max
            )
            
            if result.fun < best_val:
                h = np.zeros(N)
                h += result.x[0] / 2
                for k in range(1, K + 1):
                    h += result.x[1 + 2*(k-1)] * np.cos(2 * np.pi * k * np.arange(N) / N)
                    h += result.x[2 + 2*(k-1)] * np.sin(2 * np.pi * k * np.arange(N) / N)
                
                val = _compute_value(h)
                if val < best_val:
                    best_val = val
                    best_h = h.copy()
        except Exception:
            pass
    
    # Ensure finite and sum not zero
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-10:
        return np.ones(N)
    
    return best_h
