import numpy as np
from scipy.optimize import minimize
import time

N = 600
DELTA = (1/2) / N  # width of each interval = 1/1200

def compute_autoconvolution_max(h):
    """Compute the maximum of the autoconvolution of step function h using FFT."""
    h = np.asarray(h, dtype=float)
    n = len(h)
    
    # The autoconvolution (f*f)(t) for a step function
    # At lag m (in interval units), the value is DELTA * sum_i h[i] * h[i+m]
    # This is the autocorrelation of h, scaled by DELTA
    
    # Use FFT to compute autocorrelation efficiently
    # Pad to avoid circular convolution
    size = 2 * n
    f = np.zeros(size)
    f[:n] = h
    
    F = np.fft.fft(f)
    autocorr = np.fft.ifft(F * np.conj(F)).real
    autocorr = autocorr[:n]  # take first n lags (non-negative)
    
    # The autocorr[m] = sum_i h[i] * h[i+m] for m=0,1,...,n-1
    # Wait, actually ifft(|F|^2) gives circular autocorrelation. 
    # For linear autocorrelation, we need to be careful.
    
    # Actually, the standard way: 
    # autocorr[m] = sum_{i=0}^{n-m-1} h[i] * h[i+m]
    # This is what we get from the FFT approach if we pad properly
    
    max_val = np.max(autocorr) * DELTA
    return max_val

def objective(h):
    """Minimize max(autoconvolution) / (integral)^2"""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0)
    integral = np.sum(h) * DELTA
    if integral < 1e-10:
        return 1e10
    max_ac = compute_autoconvolution_max(h)
    return max_ac / (integral * integral)

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_h = np.ones(N)
    best_score = objective(best_h)
    
    bounds = [(0, None)] * N
    np.random.seed(42)
    
    for restart in range(5):
        if time.time() > deadline:
            break
        
        if restart == 0:
            x0 = np.ones(N)
        elif restart == 1:
            x0 = np.random.exponential(1.0, N)
        elif restart == 2:
            # Cosine-shaped
            x0 = np.cos(np.linspace(0, np.pi, N)) ** 2
            x0 = np.maximum(x0, 0)
        elif restart == 3:
            # Two bumps
            x0 = np.zeros(N)
            x0[:N//4] = 1.0
            x0[3*N//4:] = 1.0
        else:
            x0 = np.random.dirichlet(np.ones(N)) * N
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', 
                             bounds=bounds, options={'maxiter': 100, 'maxfun': 300})
            
            if time.time() > deadline:
                break
            
            if result.fun < best_score:
                best_score = result.fun
                best_h = np.maximum(result.x, 0)
        except Exception:
            continue
    
    return [float(x) for x in best_h]
