"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 600
# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Initial guess: uniform distribution
    x0 = np.ones(N)
    
    # Define the objective function
    def objective(x):
        # x is a vector of nonnegative step heights
        # Compute autoconvolution using FFT
        # The autoconvolution of a step function with N intervals can be computed efficiently
        # using FFT. The convolution of two step functions with N intervals has 2N-1 intervals,
        # but we only need the maximum value.
        
        # For step functions on equal intervals, the autoconvolution at point k is:
        # sum_{i=0}^{N-1} x[i] * x[j] where |i-j| = k
        # This is the circular autocorrelation, but we need linear autocorrelation
        
        # Using FFT for linear convolution:
        # Pad to avoid circular convolution effects
        n_fft = 2 * N
        x_pad = np.zeros(n_fft)
        x_pad[:N] = x
        
        # Compute FFT
        X = np.fft.fft(x_pad)
        # Autoconvolution is |X|^2 in frequency domain
        conv = np.fft.ifft(np.abs(X)**2).real
        
        # The linear autoconvolution is in the first N elements and last N-1 elements
        # Actually, for linear convolution of two length-N sequences, the result has length 2N-1
        # conv[0:N] gives the first part, and conv[N-1:2*N-1] gives the overlapping part
        
        # The maximum of the autoconvolution is max(conv)
        # But we need to normalize. The task says "normalized maximum"
        # Normalization: typically divide by the square of the integral (or sum for step functions)
        # For step functions with equal interval width dx = (1/4 - (-1/4))/N = 0.5/N
        # The integral is sum(x) * dx
        # The autoconvolution integral is (sum(x) * dx)^2
        # So normalized max = max(conv) / (sum(x) * dx)^2
        
        s = np.sum(x)
        dx = 0.5 / N
        norm = (s * dx)**2
        if norm < 1e-15:
            return 1e15  # Avoid division by zero
        
        return np.max(conv) / norm
    
    # Bounds: x[i] >= 0
    bounds = [(0, None)] * N
    
    # Use L-BFGS-B which handles bounds well
    best_x = x0.copy()
    best_obj = objective(x0)
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                         options={'maxiter': 5000, 'maxfun': 50000, 'gtol': 1e-8})
        if result.success or result.fun < best_obj:
            best_x = result.x
            best_obj = result.fun
    except Exception:
        pass
    
    # Check if we're still within time limit
    if time.time() > deadline:
        return list(best_x)
    
    # Try to improve with local search if time permits
    # Use a simple coordinate descent approach
    step_size = 0.1
    for _ in range(100):  # Limited iterations
        if time.time() > deadline - 5.0:
            break
        improved = False
        for i in range(N):
            if time.time() > deadline - 5.0:
                break
            
            # Try increasing
            x_trial = best_x.copy()
            x_trial[i] += step_size
            obj_trial = objective(x_trial)
            if obj_trial < best_obj:
                best_x = x_trial
                best_obj = obj_trial
                improved = True
            
            # Try decreasing
            if best_x[i] > step_size:
                x_trial = best_x.copy()
                x_trial[i] -= step_size
                obj_trial = objective(x_trial)
                if obj_trial < best_obj:
                    best_x = x_trial
                    best_obj = obj_trial
                    improved = True
        
        if not improved:
            break
    
    # Ensure nonnegativity
    best_x = np.maximum(best_x, 0)
    
    return list(best_x)
# EVOLVE-BLOCK-END
