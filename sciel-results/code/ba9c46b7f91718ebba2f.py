import numpy as np
from scipy.optimize import minimize
import time

N = 600
h = (1/4 - (-1/4)) / N  # interval width = 1/1200

def compute_score(heights):
    """Compute the normalized maximum of the autoconvolution."""
    heights = np.asarray(heights, dtype=np.float64)
    if np.any(heights < 0):
        return 1e10
    # Autoconvolution of step function
    # (f*f)(t) for step function: use FFT or direct
    # For efficiency, use FFT-based convolution
    # Pad to avoid circular convolution
    n_fft = 2 * N
    f_padded = np.zeros(n_fft)
    f_padded[:N] = heights
    conv = np.fft.ifft(np.fft.fft(f_padded) ** 2).real * h
    # The autoconvolution is defined on [-1/2, 1/2] effectively
    # We need the maximum over the valid range
    # The convolution result has length n_fft, representing t from 0 to n_fft*h
    # But autoconvolution (f*f)(t) = ∫f(τ)f(t-τ)dτ for t in [-1/2, 1/2]
    # Actually for our setup, the relevant range is t in [-1/2, 1/2]
    # The linear convolution of two signals of length N has length 2N-1
    # Let me use direct linear convolution
    conv_lin = np.convolve(heights, heights, mode='full') * h
    # conv_lin[i] corresponds to t = (i - (N-1)) * h for i=0..2N-2
    # The maximum should be near the center
    max_val = np.max(conv_lin)
    # Normalization: (∫f)² = (sum(heights)*h)²
    integral = np.sum(heights) * h
    if integral <= 0:
        return 1e10
    return max_val / (integral * integral)

def construct():
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Start with uniform
    x0 = np.ones(N)
    best_heights = x0.copy()
    best_score = compute_score(x0)
    
    # Try a few optimization strategies
    # Strategy 1: scipy L-BFGS-B with bounds
    try:
        def objective(x):
            if time.time() > deadline - 5:
                return best_score  # stop early
            return compute_score(x)
        
        res = minimize(objective, x0, method='L-BFGS-B', 
                      bounds=[(0, None)] * N,
                      options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.fun < best_score:
            best_score = res.fun
            best_heights = res.x
    except Exception:
        pass
    
    if time.time() > deadline - 10:
        return list(best_heights)
    
    # Strategy 2: Try structured perturbations
    # Try a quadratic profile: higher in middle
    for trial in range(20):
        if time.time() > deadline - 10:
            break
        # Various profiles
        t = np.linspace(-1, 1, N)
        if trial == 0:
            profile = 1 + 0.5 * (1 - t**2)  # parabolic
        elif trial == 1:
            profile = 1 + 0.3 * np.cos(np.pi * t)
        elif trial == 2:
            profile = np.exp(-0.5 * t**2)  # Gaussian-like
        else:
            profile = 1 + 0.2 * np.sin(np.pi * t * (trial - 2))
        
        profile = np.maximum(profile, 0)
        if np.sum(profile) == 0:
            continue
        score = compute_score(profile)
        if score < best_score:
            best_score = score
            best_heights = profile.copy()
    
    return list(best_heights)
