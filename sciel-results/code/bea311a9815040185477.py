"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 600

def _autoconv_peak(x):
    """Compute the maximum value of the autoconvolution of x.
    
    The autoconvolution at lag k is sum_{i} x[i]*x[i+k].
    We compute this efficiently using FFT.
    """
    # Zero-pad to avoid circular convolution
    size = 2 * N
    x_padded = np.zeros(size)
    x_padded[:N] = x
    # FFT-based convolution: x * x (autoconvolution)
    fx = np.fft.rfft(x_padded)
    conv = np.fft.irfft(fx * fx, n=size)
    # The autoconvolution values are conv[0:N] (lags 0 to N-1)
    # Actually, for linear convolution of two N-length sequences, 
    # the result has length 2N-1, and lag k corresponds to index k in the 
    # "toeplitz" sense. Let me reconsider.
    # 
    # The autoconvolution (correlation-like) at lag k is sum_i x[i]*x[i+k]
    # for k >= 0. This is the cross-correlation, not convolution.
    # 
    # For autoconvolution in the sense of (f*f)(t) = int f(t-s)f(s)ds,
    # on a discrete grid, this is the convolution. The peak of f*f is what matters.
    # 
    # Let's use the standard convolution: (x * x)[k] = sum_i x[i]*x[k-i]
    # For nonnegative x, the peak will be at k around N-1 (center).
    # 
    # Using FFT: the linear convolution of two N-length sequences can be computed
    # by zero-padding to length >= 2N-1, FFT, multiply, IFFT.
    size = 2 * N - 1
    x_pad = np.zeros(size)
    x_pad[:N] = x
    fx = np.fft.rfft(x_pad)
    conv = np.fft.irfft(fx * fx, n=size)
    return np.max(conv)

def _score(x):
    """Normalized maximum of autoconvolution.
    
    For a step function with equal intervals on [-1/4, 1/4], the function value
    at interval i is x[i]. The autoconvolution integral is approximately
    sum_{i,j} x[i]*x[j] * (overlap of intervals i and j).
    
    For equal intervals of width w = (1/2)/600 = 1/1200 on [-1/4, 1/4],
    the autoconvolution (f*f)(t) at time t is int f(t-s)f(s)ds.
    
    On a discrete grid, if we think of f as piecewise constant with value x[i]
    on interval [a_i, a_i + w], then (f*f)(t) = sum_i x[i] * (measure of s such that
    s in [a_i, a_i+w] and t-s in [a_j, a_j+w] for some j)... this is getting complex.
    
    A simpler model: the normalized peak autoconvolution is approximately
    max_k (sum_i x[i]*x[i+k]) / (sum_i x[i])^2 * N
    or some normalization. Let me use a standard normalization.
    
    Actually, for the autoconvolution of a step function, the peak value scales
    with the square of the total mass times the interval width. Let's use:
    score = max(autoconv) / (sum(x))^2
    which is a normalized measure.
    """
    total_mass = np.sum(x)
    if total_mass < 1e-15:
        return np.inf
    peak = _autoconv_peak(x)
    return peak / (total_mass ** 2)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    # Initialize with random binary pattern (50% on)
    np.random.seed(42)
    x = np.random.randint(0, 2, size=N).astype(float)
    
    # Compute initial score
    current_score = _score(x)
    best_x = x.copy()
    best_score = current_score
    
    # Simulated annealing parameters
    T = 1e-3
    cooling_rate = 0.9995
    min_T = 1e-8
    
    iteration = 0
    while time.time() < deadline and T > min_T:
        iteration += 1
        
        # Pick a random interval to flip
        idx = np.random.randint(N)
        
        # Try flipping the bit
        x[idx] = 1.0 - x[idx]
        
        # Compute new score
        new_score = _score(x)
        
        # Accept or reject
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_x = x.copy()
        else:
            delta = new_score - current_score
            if np.random.rand() < np.exp(-delta / (T + 1e-15)):
                current_score = new_score
            else:
                # Reject: revert the flip
                x[idx] = 1.0 - x[idx]
        
        # Cool down
        T *= cooling_rate
    
    return best_x.tolist()
# EVOLVE-BLOCK-END
