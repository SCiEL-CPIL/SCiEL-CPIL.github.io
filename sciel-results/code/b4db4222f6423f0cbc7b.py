"""
Optimized program for the suite task.
Strategy: Use a Gaussian mixture model parameterization with L-BFGS-B optimization
to minimize the normalized maximum of autoconvolution. The autoconvolution of a
Gaussian is a wider Gaussian, and mixing Gaussians allows shaping the peak.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import softmax

N = 600

def _compute_score(heights):
    """Compute the normalized maximum of autoconvolution."""
    if len(heights) == 0:
        return np.inf
    h = np.asarray(heights, dtype=np.float64)
    s = h.sum()
    if s <= 0:
        return np.inf
    # Autoconvolution via FFT
    n = len(h)
    fft_size = 2 * n
    H = np.fft.rfft(h, n=fft_size)
    auto = np.fft.irfft(H * np.conj(H), n=fft_size)[:n]
    max_val = np.max(auto)
    return max_val / (s * s)

def _gaussian_basis(x, centers, widths):
    """Compute sum of Gaussians at points x given centers and widths."""
    result = np.zeros_like(x)
    for c, w in zip(centers, widths):
        result += np.exp(-0.5 * ((x - c) / w) ** 2) / (w * np.sqrt(2 * np.pi))
    return result

def _parametric_to_step(params, n_gaussians):
    """Convert parametric representation to step heights."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.5 / N  # centers of intervals
    centers = params[:n_gaussians]
    widths = np.exp(params[n_gaussians:2 * n_gaussians])
    weights = np.exp(params[2 * n_gaussians:3 * n_gaussians])
    
    heights = np.zeros(N)
    for i in range(n_gaussians):
        heights += weights[i] * np.exp(-0.5 * ((x - centers[i]) / widths[i]) ** 2) / (widths[i] * np.sqrt(2 * np.pi))
    
    # Normalize to sum to 1
    s = heights.sum()
    if s > 0:
        heights /= s
    else:
        heights = np.ones(N) / N
    
    return heights

def _objective(params, n_gaussians):
    """Objective function for optimization."""
    heights = _parametric_to_step(params, n_gaussians)
    return _compute_score(heights)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Start with uniform as baseline
    uniform_heights = np.ones(N)
    best_heights = uniform_heights.copy()
    best_score = _compute_score(best_heights)
    
    # Try different numbers of Gaussians
    for n_gaussians in [1, 2, 3, 4, 5, 6, 8]:
        if time.time() > deadline - 20:
            break
        
        # Initialize with spread out Gaussians
        np.random.seed(42 + n_gaussians)
        
        # Multiple restarts
        n_restarts = 5
        for restart in range(n_restarts):
            if time.time() > deadline - 15:
                break
            
            # Initialize parameters
            centers = np.linspace(-0.2, 0.2, n_gaussians) + np.random.randn(n_gaussians) * 0.05
            widths = np.exp(np.random.randn(n_gaussians) * 0.5 + np.log(0.05))
            weights = np.exp(np.random.randn(n_gaussians) * 0.3)
            
            params = np.concatenate([centers, np.log(widths), np.log(weights)])
            
            # Optimize
            try:
                result = minimize(
                    _objective,
                    params,
                    args=(n_gaussians,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if result.success or result.fun < best_score:
                    heights = _parametric_to_step(result.x, n_gaussians)
                    score = _compute_score(heights)
                    if score < best_score:
                        best_score = score
                        best_heights = heights.copy()
            except Exception:
                pass
    
    # Return best heights found
    return list(best_heights)

# EVOLVE-BLOCK-END
