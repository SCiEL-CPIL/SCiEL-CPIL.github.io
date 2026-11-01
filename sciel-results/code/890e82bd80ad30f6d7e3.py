import numpy as np
from scipy.optimize import minimize
import time

N = 400
DEADLINE = 140.0  # seconds, safety margin before 150s limit

def _autoconvolution(h):
    """Compute autoconvolution of step function heights using FFT."""
    n = 2 * N
    H = np.fft.rfft(h, n=n)
    conv = np.fft.irfft(H * H, n=n)
    return conv

def _c3_bound(h):
    """Compute C3 upper bound: L-infinity norm of autoconvolution normalized by N."""
    conv = _autoconvolution(h)
    return np.max(np.abs(conv)) / N

def construct():
    start_time = time.time()
    
    # Start with a random initialization
    np.random.seed(42)
    h0 = np.random.choice([-1.0, 1.0], size=N)
    
    best_h = h0.copy()
    best_val = _c3_bound(best_h)
    
    # Try multiple random restarts
    for restart in range(20):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # New random initialization
        h0 = np.random.choice([-1.0, 1.0], size=N)
        val0 = _c3_bound(h0)
        
        # Use L-BFGS-B to optimize
        def objective(x):
            return _c3_bound(x)
        
        result = minimize(objective, h0, method='L-BFGS-B', 
                         options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-8})
        
        if result.fun < best_val:
            best_val = result.fun
            best_h = result.x.copy()
        
        if time.time() - start_time > DEADLINE - 5:
            break
    
    return best_h
