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
    m = np.max(conv)
    return m + np.log(np.sum(np.exp(conv - m)) / len(conv)) / beta

def _reconstruct_h(params, K):
    """Reconstruct h from Fourier coefficients."""
    h = np.zeros(N)
    t = np.arange(N)
    h += params[0] / 2
    for k in range(1, K + 1):
        h += params[1 + 2*(k-1)] * np.cos(2 * np.pi * k * t / N)
        h += params[2 + 2*(k-1)] * np.sin(2 * np.pi * k * t / N)
    return h

def _objective(params, K, beta=50.0):
    """Objective: smooth max of autocorrelation, normalized."""
    h = _reconstruct_h(params, K)
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h)
    sm = _smooth_max_surrogate(conv, beta)
    return sm / (s ** 2)

def construct():
    """Construct step heights minimizing autocorrelation peak."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Use more Fourier modes for better representation
    K = 20  # 20 cosine + 20 sine + 1 DC = 41 parameters
    
    best_h = np.ones(N)
    best_val = _compute_value(best_h)
    
    # Multi-start optimization with diverse initializations
    np.random.seed(42)
    n_starts = 30
    
    for i in range(n_starts):
        if time.time() > deadline:
            break
        
        # Initialize with random Fourier coefficients
        if i == 0:
            # Start near all-ones (seed)
            params0 = np.zeros(1 + 2*K)
            params0[0] = N  # DC term to make sum = N
        elif i == 1:
            # Start with a single dominant frequency
            params0 = np.zeros(1 + 2*K)
            params0[0] = N
            params0[1] = N * 0.1  # First cosine mode
        elif i == 2:
            # Start with alternating pattern
            params0 = np.zeros(1 + 2*K)
            params0[0] = N
            params0[1 + 2*(K-1)] = N * 0.05  # High frequency
        else:
            # Random initialization with some structure
            params0 = np.random.randn(1 + 2*K) * 0.05
            params0[0] = N * np.random.uniform(0.8, 1.2)
        
        # Phase 1: Coarse optimization with moderate beta
        try:
            result = minimize(
                _objective,
                params0,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10},
                args=(K, 30.0)
            )
            
            if result.success or result.fun < best_val:
                h = _reconstruct_h(result.x, K)
                val = _compute_value(h)
                if val < best_val:
                    best_val = val
                    best_h = h.copy()
                    
        except Exception:
            continue
    
    # Phase 2: Fine optimization with high beta on best solution
    if time.time() < deadline - 30 and best_val < 2.0:
        h = best_h
        # Extract current Fourier coefficients
        params_current = np.zeros(1 + 2*K)
        t = np.arange(N)
        params_current[0] = 2 * np.sum(h) / N
        for k in range(1, K + 1):
            params_current[1 + 2*(k-1)] = 2 * np.sum(h * np.cos(2 * np.pi * k * t / N)) / N
            params_current[2 + 2*(k-1)] = 2 * np.sum(h * np.sin(2 * np.pi * k * t / N)) / N
        
        try:
            result = minimize(
                _objective,
                params_current,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12},
                args=(K, 100.0)  # Higher beta for sharper max
            )
            
            if result.fun < best_val:
                h = _reconstruct_h(result.x, K)
                val = _compute_value(h)
                if val < best_val:
                    best_val = val
                    best_h = h.copy()
        except Exception:
            pass
    
    # Phase 3: Additional restarts from best solution with perturbations
    if time.time() < deadline - 15 and best_val < 2.0:
        for i in range(10):
            if time.time() > deadline:
                break
            
            # Perturb the best solution
            params_perturbed = np.zeros(1 + 2*K)
            t = np.arange(N)
            params_perturbed[0] = 2 * np.sum(best_h) / N
            for k in range(1, K + 1):
                params_perturbed[1 + 2*(k-1)] = 2 * np.sum(best_h * np.cos(2 * np.pi * k * t / N)) / N
                params_perturbed[2 + 2*(k-1)] = 2 * np.sum(best_h * np.sin(2 * np.pi * k * t / N)) / N
            
            # Add small random perturbation
            params_perturbed += np.random.randn(1 + 2*K) * 0.01
            params_perturbed[0] = 2 * np.sum(best_h) / N  # Keep DC term
            
            try:
                result = minimize(
                    _objective,
                    params_perturbed,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-13, 'gtol': 1e-11},
                    args=(K, 80.0)
                )
                
                if result.fun < best_val:
                    h = _reconstruct_h(result.x, K)
                    val = _compute_value(h)
                    if val < best_val:
                        best_val = val
                        best_h = h.copy()
            except Exception:
                continue
    
    # Ensure finite and sum not zero
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-10:
        return np.ones(N)
    
    return best_h
