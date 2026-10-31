import numpy as np
from scipy.optimize import minimize
import time

N = 600
DEADLINE = 130.0  # safety margin before 150s kill

def compute_autoconv_max(heights):
    """Compute the normalized maximum of the autoconvolution."""
    # heights: array of N nonnegative values
    # Autoconvolution via FFT
    # Pad to at least 2*N-1 for linear convolution
    nfft = 1 << (2 * N - 1).bit_length()
    f = np.zeros(nfft)
    f[:N] = heights
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=nfft)
    # The autoconvolution is conv[:2*N-1]
    conv = conv[:2 * N - 1]
    max_val = np.max(conv)
    # Normalization: divide by (sum of heights)^2 / N? or (mean)^2 * N?
    # For uniform heights=1: conv peak = N (at center), sum=N, mean=1
    # If normalized by (sum/N)^2 * N = 1 * N = N, then uniform gives 1.0
    # Actually for uniform: conv_k = N - |k| for |k|<N, peak = N
    # (sum h)^2 / N = N^2 / N = N, so max_conv / ((sum h)^2 / N) = N/N = 1
    total = np.sum(heights)
    if total == 0:
        return 0.0
    normalized_max = max_val / (total * total / N)
    return normalized_max

def construct():
    start_time = time.time()
    
    # Start with uniform (baseline score 1.0)
    best_heights = np.ones(N)
    best_score = compute_autoconv_max(best_heights)
    
    # Parameterize as cosine series: f_i = a0 + sum_{k=1}^{K} a_k * cos(2*pi*k*i/N)
    # This gives periodic/symmetric functions. Use enough terms for flexibility.
    K = 30  # number of cosine terms
    
    def decode(params):
        """Decode parameters to nonnegative heights."""
        a0 = params[0]
        coeffs = params[1:]
        i = np.arange(N)
        heights = np.full(N, a0)
        for k in range(1, K + 1):
            if k <= len(coeffs):
                heights += coeffs[k-1] * np.cos(2 * np.pi * k * i / N)
        # Enforce nonnegativity
        heights = np.maximum(heights, 0.0)
        return heights
    
    def objective(params):
        heights = decode(params)
        if np.all(heights == 0):
            return 1e10
        return compute_autoconv_max(heights)
    
    # Initial parameters: a0=1, all other coeffs 0
    x0 = np.zeros(K + 1)
    x0[0] = 1.0
    
    # Try L-BFGS-B with multiple restarts
    best_params = x0.copy()
    
    # First, try a single optimization from uniform
    try:
        result = minimize(objective, x0, method='L-BFGS-B', 
                         options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
        if result.fun < best_score:
            best_score = result.fun
            best_params = result.x
            best_heights = decode(result.x)
    except Exception:
        pass
    
    # Check time
    if time.time() - start_time > DEADLINE - 30:
        return best_heights.tolist()
    
    # Try multiple random restarts with small perturbations
    np.random.seed(42)
    n_restarts = 20
    for r in range(n_restarts):
        if time.time() - start_time > DEADLINE - 15:
            break
        
        # Random small perturbation of the best found
        if r == 0:
            x_start = x0.copy()
        else:
            x_start = best_params + np.random.randn(K + 1) * 0.1
        
        try:
            result = minimize(objective, x_start, method='L-BFGS-B',
                             options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x
                best_heights = decode(result.x)
        except Exception:
            pass
    
    # Final polish with tighter tolerance
    if time.time() - start_time < DEADLINE - 10:
        try:
            result = minimize(objective, best_params, method='L-BFGS-B',
                             options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
            if result.fun < best_score:
                best_score = result.fun
                best_heights = decode(result.x)
        except Exception:
            pass
    
    # Ensure nonnegativity and return
    best_heights = np.maximum(best_heights, 0.0)
    
    return best_heights.tolist()
