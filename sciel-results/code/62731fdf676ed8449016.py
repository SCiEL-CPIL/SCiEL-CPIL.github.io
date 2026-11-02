"""Improved program for the autocorrelation inequality task."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START

def _autoconv_max_abs(s):
    """Compute the maximum absolute value of the autoconvolution of s."""
    # Use FFT for efficient computation
    n_fft = 2 * N
    f = np.fft.rfft(s, n=n_fft)
    ac = np.fft.irfft(f * np.conj(f), n=n_fft)
    # The autoconvolution has length 2*N-1, centered at index N-1
    # We take the full autoconvolution
    ac_full = np.fft.irfft(f * np.conj(f), n=n_fft)
    # The circular autoconvolution of length N would be different; we want linear
    # Actually, let's use a proper linear autoconvolution via FFT
    # For linear convolution, we need zero-padding to at least 2N-1
    n_fft2 = 1
    while n_fft2 < 2 * N - 1:
        n_fft2 *= 2
    f2 = np.fft.rfft(s, n=n_fft2)
    ac2 = np.fft.irfft(f2 * np.conj(f2), n=n_fft2)
    # The autoconvolution c[k] = sum_i s[i] * s[i+k] for k in [-(N-1), N-1]
    # The FFT gives us the circular convolution; with zero padding, it's linear
    # ac2[0] = sum s[i]^2, ac2[1] = sum s[i]s[i+1], etc.
    # We want the max absolute value over all lags
    return np.max(np.abs(ac2[:N]))  # positive lags 0..N-1

def _autoconv_max_abs_fast(s):
    """Faster version using precomputed FFT size."""
    n_fft = 1024  # next power of 2 >= 2*400-1 = 799
    f = np.fft.rfft(s, n=n_fft)
    ac = np.fft.irfft(f * np.conj(f), n=n_fft)
    return np.max(np.abs(ac[:N]))

def _score(s):
    """Score: lower is better. Returns the upper bound on C3."""
    return _autoconv_max_abs_fast(s)

def _flip_block(s, start, length):
    """Flip signs of a block of consecutive elements."""
    s2 = s.copy()
    s2[start:start+length] = -s2[start:start+length]
    return s2

def construct():
    """Construct a signed step function to minimize autoconvolution max abs."""
    start_time = time.time()
    deadline = 135.0  # safety margin before 150s limit
    
    # Initialize with a sine-wave-derived sign pattern
    # This provides a smooth transition between positive and negative blocks
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    # Low frequency sine to create large positive/negative blocks
    freq = 1.0
    phase = 0.0
    s = np.sign(np.sin(2 * np.pi * freq * x + phase))
    # Replace zeros with 1
    s[s == 0] = 1
    
    best_s = s.copy()
    best_score = _score(best_s)
    
    # Try several initial configurations
    for freq in [0.5, 1.0, 1.5, 2.0, 3.0]:
        for phase in [0.0, np.pi/4, np.pi/2, 3*np.pi/4]:
            if time.time() > deadline:
                break
            x = np.linspace(-0.25, 0.25, N, endpoint=False)
            s_try = np.sign(np.sin(2 * np.pi * freq * x + phase))
            s_try[s_try == 0] = 1
            sc = _score(s_try)
            if sc < best_score:
                best_score = sc
                best_s = s_try.copy()
    
    # Simulated annealing with block flips
    current_s = best_s.copy()
    current_score = best_score
    
    # Precompute FFT size
    n_fft = 1024
    
    def _eval(s):
        f = np.fft.rfft(s, n=n_fft)
        ac = np.fft.irfft(f * np.conj(f), n=n_fft)
        return np.max(np.abs(ac[:N]))
    
    # Simulated annealing parameters
    T_start = 0.1
    T_end = 0.001
    n_iterations = 50000
    cool_rate = (T_end / T_start) ** (1.0 / max(n_iterations, 1))
    
    T = T_start
    for it in range(n_iterations):
        if time.time() > deadline:
            break
        
        # Random block flip
        block_size = np.random.randint(2, 33)  # 2 to 32
        start = np.random.randint(0, N - block_size + 1)
        
        new_s = current_s.copy()
        new_s[start:start+block_size] = -new_s[start:start+block_size]
        
        new_score = _eval(new_s)
        delta = new_score - current_score
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_s = new_s
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_s = current_s.copy()
        
        T *= cool_rate
    
    return best_s

# EVOLVE-BLOCK-END
