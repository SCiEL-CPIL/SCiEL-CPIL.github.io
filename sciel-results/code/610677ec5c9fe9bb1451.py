import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 600
DELTA_T = (1.0/4 - (-1.0/4)) / N  # = 1/1200

def _autoconv_max_ratio(h):
    """
    Compute the normalized maximum of the autoconvolution.
    The score is max_k (autoconv_k) / (sum_i h_i)^2
    """
    # Ensure nonnegativity
    h = np.maximum(h, 0)
    
    if np.all(h == 0):
        return 0.0  # degenerate case
    
    # Compute autoconvolution via FFT
    # We need to handle the circular vs linear convolution carefully
    # For step functions on equal intervals, the autoconvolution at discrete points
    # can be computed as a linear convolution
    
    # Use FFT for efficiency
    n_fft = 2 * N
    H = np.fft.rfft(h, n=n_fft)
    autoconv = np.fft.irfft(H * H, n=n_fft)
    
    # The autoconvolution values are at positions 0, 1, ..., 2N-1
    # But we only care about the valid range: 0 to 2N-1 (all are valid for linear conv)
    # However, for the problem, we need the autoconvolution on the domain
    
    # The maximum of the autoconvolution
    max_autoconv = np.max(autoconv)
    
    # Normalization: (sum h_i * delta_t)^2
    integral = np.sum(h) * DELTA_T
    norm = integral ** 2
    
    if norm < 1e-30:
        return 0.0
    
    return max_autoconv / norm

def _objective(x):
    """Objective function for optimization."""
    h = x
    return _autoconv_max_ratio(h)

def construct():
    """
    Find a nonnegative step function with 600 equal intervals on [-1/4, 1/4]
    to minimize the normalized maximum of its autoconvolution.
    """
    start_time = time.time()
    deadline = 120.0  # seconds
    
    # Start with uniform distribution
    h_best = np.ones(N)
    score_best = _autoconv_max_ratio(h_best)
    
    # Try a few different initializations and optimization strategies
    initializations = [
        np.ones(N),  # uniform
        np.exp(-50 * (np.linspace(-0.25, 0.25, N))**2),  # Gaussian-like
        np.ones(N) * (1 + 0.1 * np.random.randn(N)),  # perturbed uniform
    ]
    
    for init in initializations:
        if time.time() > deadline - 30:
            break
        
        init = np.maximum(init, 0)
        
        # First, try a simple gradient-based optimization with log-barrier for nonnegativity
        # Use log-transform to enforce positivity
        x0 = np.log(init + 1e-10)
        
        def log_obj(x):
            h = np.exp(x)
            return _autoconv_max_ratio(h)
        
        try:
            result = minimize(
                log_obj,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'gtol': 1e-6}
            )
            
            h_candidate = np.exp(result.x)
            score_candidate = _autoconv_max_ratio(h_candidate)
            
            if score_candidate < score_best:
                score_best = score_candidate
                h_best = h_candidate.copy()
        except:
            pass
        
        # Also try differential evolution for global optimization on smaller problems
        if time.time() > deadline - 20:
            break
        
        # Reduce dimensionality for DE by optimizing a parameterized family
        # Use a mixture of Gaussians parameterized by a few parameters
        def param_obj(params):
            # params: [centers, widths, amplitudes]
            h = np.zeros(N)
            x_grid = np.linspace(-0.25, 0.25, N)
            for i in range(len(params) // 3):
                c = params[3*i]
                w = params[3*i+1]
                a = params[3*i+2]
                h += a * np.exp(-((x_grid - c)**2) / (2 * w**2))
            h = np.maximum(h, 0)
            return _autoconv_max_ratio(h)
        
        # Use 2 Gaussians: 6 parameters
        bounds = [(-0.2, 0.2), (0.01, 0.15), (0.1, 10)] * 2
        try:
            de_result = differential_evolution(
                param_obj,
                bounds,
                maxiter=50,
                popsize=10,
                seed=42,
                tol=1e-4,
                polish=True
            )
            
            # Reconstruct h from best params
            params = de_result.x
            h = np.zeros(N)
            x_grid = np.linspace(-0.25, 0.25, N)
            for i in range(len(params) // 3):
                c = params[3*i]
                w = params[3*i+1]
                a = params[3*i+2]
                h += a * np.exp(-((x_grid - c)**2) / (2 * w**2))
            h = np.maximum(h, 0)
            
            score_candidate = _autoconv_max_ratio(h)
            if score_candidate < score_best:
                score_best = score_candidate
                h_best = h.copy()
        except:
            pass
    
    # Final: ensure we return a valid nonnegative array
    h_best = np.maximum(h_best, 0)
    
    # If all zeros, fall back to uniform
    if np.all(h_best == 0):
        h_best = np.ones(N)
    
    return [float(x) for x in h_best]
