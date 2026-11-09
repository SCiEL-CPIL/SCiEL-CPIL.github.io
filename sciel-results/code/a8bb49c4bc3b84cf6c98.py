import numpy as np
import time
from scipy.optimize import minimize

N = 50
START_TIME = time.time()
TIME_LIMIT = 140  # Safety margin before 150s kill

def compute_ratio(heights):
    """Compute the ratio ||f||_2^2 / (||f||_1 * sup|f*f|) for non-negative step heights."""
    a = np.asarray(heights, dtype=np.float64)
    if np.any(a < 0):
        a = np.maximum(a, 0)
    dx = 0.5 / N  # interval length / N
    
    norm1 = np.sum(a) * dx
    if norm1 == 0:
        return 0.0
    
    # Compute autoconvolution of step function
    # f is a step function: f(x) = a[i] for x in [i*dx - 0.25, (i+1)*dx - 0.25)
    # (f*f)(t) = integral f(x)f(t-x) dx
    # For step functions, this can be computed via FFT or direct convolution
    # Using FFT for efficiency
    n = N
    # Pad for linear convolution
    conv_len = 2 * n - 1
    f_padded = np.zeros(conv_len)
    f_padded[:n] = a
    f_conv = np.fft.irfft(np.fft.rfft(f_padded, n=2*conv_len) ** 2)[:conv_len] * dx
    
    # The autoconvolution is a piecewise linear function
    # sup |f*f| for non-negative f is the maximum of f_conv
    sup_conv = np.max(f_conv)
    
    if sup_conv == 0:
        return 0.0
    
    norm2_sq = np.sum(a**2) * dx
    ratio = norm2_sq / (norm1 * sup_conv)
    return ratio

def neg_ratio(x):
    """Negative ratio for minimization."""
    return -compute_ratio(x)

def construct():
    """Optimize height profile to maximize the autocorrelation ratio."""
    deadline = START_TIME + TIME_LIMIT
    
    # Start with uniform initialization
    x0 = np.ones(N)
    
    # Try multiple initializations and local optimizations
    best_ratio = -np.inf
    best_heights = x0.copy()
    
    # Different initialization strategies
    initializations = [
        np.ones(N),
        np.exp(-np.linspace(-3, 3, N)**2 / 2),  # Gaussian-like
        np.abs(np.sin(np.linspace(0, np.pi, N))),  # Sine bump
        np.concatenate([np.ones(N//2), np.zeros(N - N//2)]),  # Half-on
        np.linspace(1, 0.1, N),  # Decaying
        np.concatenate([np.linspace(0.1, 1, N//2), np.linspace(1, 0.1, N - N//2)]),  # Peak in middle
    ]
    
    # Add random initializations
    rng = np.random.default_rng(42)
    for _ in range(10):
        init = rng.random(N)
        init[init < 0.05] = 0  # Some zeros to allow sparse structures
        initializations.append(init)
    
    for init in initializations:
        if time.time() > deadline - 10:
            break
        
        init = np.maximum(init, 0)
        if np.sum(init) == 0:
            continue
        
        # Normalize to avoid scaling issues during optimization
        init_norm = init / np.sum(init) * N  # Mean height = 1
        
        try:
            result = minimize(
                neg_ratio,
                init_norm,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 200, 'maxfun': 500}
            )
            
            if result.success or result.fun < -best_ratio:
                r = -result.fun
                if r > best_ratio:
                    best_ratio = r
                    best_heights = result.x
        except Exception:
            continue
        
        # Also try Nelder-Mead for some initializations
        if time.time() > deadline - 10:
            break
        
        try:
            result2 = minimize(
                neg_ratio,
                init_norm,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if -result2.fun > best_ratio:
                best_ratio = -result2.fun
                best_heights = result2.x
        except Exception:
            continue
    
    # Ensure non-negative and return
    best_heights = np.maximum(best_heights, 0)
    
    # If optimization failed, fall back to uniform
    if best_ratio <= 0 or np.all(best_heights == 0):
        return [1.0] * N
    
    # Scale heights to a reasonable magnitude (ratio is scale-dependent)
    # Use moderate scale to avoid numerical issues
    scale = 1.0 / max(np.max(best_heights), 1e-10)
    heights = best_heights * scale
    
    return list(heights)
