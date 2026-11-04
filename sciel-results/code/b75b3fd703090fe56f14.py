"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _thue_morse(n):
    """Generate Thue-Morse sequence of length n, mapped to +/-1."""
    # Thue-Morse: t_n = parity of number of 1s in binary representation of n
    # Use bitwise trick: t_n = popcount(n) % 2
    # For n up to 400, we can compute directly
    seq = np.zeros(n, dtype=np.int8)
    for i in range(n):
        # Count bits set in i
        seq[i] = bin(i).count('1') % 2
    return 2 * seq.astype(np.float64) - 1


def _autoconvolution_peak(x):
    """Compute the peak absolute value of the autoconvolution."""
    # Linear autoconvolution via FFT for speed
    n = len(x)
    # Use next power of 2 >= 2n-1 for efficient FFT
    fft_size = 1 << (2 * n - 1).bit_length()
    X = np.fft.rfft(x, fft_size)
    conv = np.fft.irfft(X * X, fft_size)
    # Linear autoconvolution is the first 2n-1 values
    conv = conv[:2 * n - 1]
    return np.max(np.abs(conv))


def _local_search(x, time_limit=10):
    """Local search: flip individual bits to reduce autoconvolution peak."""
    x = x.copy()
    best_peak = _autoconvolution_peak(x)
    start_time = time.time()
    n = len(x)
    
    # Precompute indices
    indices = np.arange(n)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in indices:
            if (time.time() - start_time) > time_limit:
                break
            # Try flipping position i
            x[i] *= -1
            new_peak = _autoconvolution_peak(x)
            if new_peak < best_peak - 1e-12:
                best_peak = new_peak
                improved = True
            else:
                x[i] *= -1  # revert
    return x, best_peak


def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    best_x = None
    best_peak = np.inf
    
    # Start with Thue-Morse as primary initialization
    # It has known low autocorrelation properties
    x_tm = _thue_morse(N)
    x_refined, peak = _local_search(x_tm, time_limit=30)
    
    if peak < best_peak:
        best_peak = peak
        best_x = x_refined
    
    # Try a few random initializations with local search
    rng = np.random.RandomState(42)
    for restart in range(5):
        if (time.time() - start_time) > deadline - 20:
            break
        
        # Random initialization
        x_rand = rng.choice([-1.0, 1.0], size=N)
        x_refined, peak = _local_search(x_rand, time_limit=15)
        
        if peak < best_peak:
            best_peak = peak
            best_x = x_refined
    
    if best_x is None:
        # Fallback
        best_x = np.ones(N)
    
    return best_x
# EVOLVE-BLOCK-END
