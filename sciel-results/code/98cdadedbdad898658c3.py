import time
import numpy as np
from scipy.optimize import minimize

N = 600
DEADLINE = 145.0  # safety margin before 150s kill

def _time_left():
    return DEADLINE - (time.time() - _start_time)

def _autoconv_peak_ratio(heights):
    """
    Compute the normalized maximum of the autoconvolution.
    For a step function with heights h_i on N equal intervals of [-1/4, 1/4],
    the autoconvolution at lag k is sum_i h_i * h_{i-k} (with appropriate boundaries).
    The normalized value is N * max(autoconv) / (sum h)^2.
    """
    h = np.asarray(heights, dtype=np.float64)
    total = h.sum()
    if total <= 0:
        return float('inf')
    # Linear autoconvolution (not circular)
    # For step functions on equal intervals, the autoconvolution at discrete lags
    # is the linear convolution of h with itself, but we need to be careful about
    # the actual continuous autoconvolution.
    # 
    # Actually, for a step function f(x) = h_i on interval i, the autoconvolution
    # (f * f)(t) = integral f(x) f(t-x) dx. For step functions on equal intervals,
    # this is a piecewise linear function. The maximum occurs at some point.
    #
    # A simpler approach: use the discrete autoconvolution as an approximation.
    # The normalized metric is N * max(autoconv) / (sum h)^2.
    # For the uniform case, this equals 1.0.
    
    # Compute linear autoconvolution
    conv = np.convolve(h, h, mode='full')
    # The autoconvolution of a step function: each pair contributes h_i*h_j * (overlap length)
    # For equal intervals of width w = (1/4 - (-1/4))/N = (1/2)/N = 1/(2N)
    # The discrete conv approximates the integral. The actual max of the continuous
    # autoconvolution needs more care.
    
    # Let's use a more accurate computation. The step function has height h_i on 
    # interval [x_i, x_{i+1}] where x_i = -1/4 + i/N * (1/2) = -1/4 + i/(2N)
    # Width of each interval: w = 1/(2N)
    
    # The autoconvolution (f*f)(t) = integral f(x)f(t-x)dx
    # For step functions, this is a piecewise linear function with breakpoints at
    # t = x_j - x_i for all i, j.
    
    # A practical approach: evaluate the autoconvolution at many points and find max.
    # But this is expensive. Let's use the FFT-based approach for the continuous case.
    
    # Actually, let me reconsider. The task says "normalized maximum of its autoconvolution"
    # and the score is an upper bound on C1. The uniform function gives score 1.0.
    
    # For a step function with heights h_i on N equal intervals of width w=1/(2N):
    # The L1 norm is sum(h_i * w) = w * sum(h_i)
    # The autoconvolution at t=0 is integral f(x)^2 dx = w * sum(h_i^2)
    # The maximum of the autoconvolution is at least this value.
    # 
    # The normalized quantity is likely: max(f*f) / (L1(f))^2 or similar.
    # For uniform: max(f*f) = w * sum(h_i^2) = w * N * h^2 = (1/(2N)) * N * h^2 = h^2/2
    # L1(f) = w * N * h = (1/(2N)) * N * h = h/2
    # (L1)^2 = h^2/4
    # Ratio = (h^2/2) / (h^2/4) = 2. That doesn't give 1.
    
    # Let me reconsider the normalization. The score for uniform is 1.0.
    # Perhaps it's: N * max(autoconv) / (sum h)^2 where autoconv is computed discretely.
    # For uniform h_i = c: sum h = N*c, (sum h)^2 = N^2 c^2
    # Discrete autoconv at lag 0: sum h_i^2 = N c^2
    # N * N c^2 / (N^2 c^2) = 1. Yes!
    
    # So the metric is: N * max_k (sum_i h_i * h_{i+k}) / (sum_i h_i)^2
    # where the sum is over valid i for each lag k.
    
    max_conv = np.max(conv)
    return N * max_conv / (total * total)

def _cosine_parameterize(coeffs, N):
    """
    Parameterize heights as baseline + cosine series.
    coeffs[0] is baseline, coeffs[1:] are cosine coefficients.
    heights[i] = coeffs[0] + sum_{k=1}^{K} coeffs[k] * cos(2*pi*k*i/N)
    """
    i = np.arange(N)
    h = np.full(N, coeffs[0])
    for k in range(1, len(coeffs)):
        h += coeffs[k] * np.cos(2 * np.pi * k * i / N)
    return h

def _score_from_coeffs(coeffs, N):
    h = _cosine_parameterize(coeffs, N)
    # Ensure nonnegativity by clipping
    h = np.maximum(h, 0)
    return _autoconv_peak_ratio(h)

def construct():
    global _start_time
    _start_time = time.time()
    
    # Start with uniform
    best_h = [1.0] * N
    best_score = _autoconv_peak_ratio(best_h)
    
    # Try cosine series optimization with low-order terms
    # Start with a few random initializations
    K = 10  # number of cosine terms
    
    np.random.seed(42)
    
    for trial in range(20):
        if _time_left() < 5:
            break
        
        # Random initialization
        coeffs = np.zeros(K + 1)
        coeffs[0] = 1.0  # baseline
        coeffs[1:] = np.random.randn(K) * 0.1
        
        # Optimize using scipy
        result = minimize(
            _score_from_coeffs,
            coeffs,
            args=(N,),
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
        )
        
        if result.success or result.fun < best_score:
            h = _cosine_parameterize(result.x, N)
            h = np.maximum(h, 0)
            score = _autoconv_peak_ratio(h)
            if score < best_score:
                best_score = score
                best_h = h.tolist()
        
        if _time_left() < 10:
            break
    
    # Try a few more targeted initializations
    for trial in range(10):
        if _time_left() < 5:
            break
        
        # Try different baseline values
        coeffs = np.zeros(K + 1)
        coeffs[0] = 0.5 + 0.5 * trial / 10
        coeffs[1:] = np.random.randn(K) * 0.05
        
        result = minimize(
            _score_from_coeffs,
            coeffs,
            args=(N,),
            method='Nelder-Mead',
            options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8}
        )
        
        if result.fun < best_score:
            h = _cosine_parameterize(result.x, N)
            h = np.maximum(h, 0)
            score = _autoconv_peak_ratio(h)
            if score < best_score:
                best_score = score
                best_h = h.tolist()
    
    return best_h

_start_time = 0.0
