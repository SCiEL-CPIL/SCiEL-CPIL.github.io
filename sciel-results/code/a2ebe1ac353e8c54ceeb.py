"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autocorr_peak(h):
    """Compute max |autocorrelation| at non-zero lags using FFT."""
    n = len(h)
    if n <= 1:
        return 0.0
    # Zero-pad to 2*n for linear autocorrelation
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    # acf[0] is lag 0 (sum of squares), we want max |acf[k]| for k=1..n-1
    # Also consider negative lags: acf[n+1] to acf[fft_size-1] correspond to negative lags
    # But for real sequences, acf is symmetric, so max over 1..n-1 suffices
    return float(np.max(np.abs(acf[1:n])))


def _local_search(h, max_time):
    """Greedy local search: flip bits to reduce autocorrelation peak."""
    n = len(h)
    h = h.copy()
    current_peak = _autocorr_peak(h)
    start_time = time.time()
    
    # Precompute positions
    positions = np.arange(n)
    
    while time.time() - start_time < max_time:
        improved = False
        # Shuffle positions to avoid systematic bias
        np.random.shuffle(positions)
        
        for i in positions:
            if time.time() - start_time > max_time:
                break
            # Try flipping position i
            old_val = h[i]
            h[i] = -h[i]
            new_peak = _autocorr_peak(h)
            if new_peak < current_peak:
                current_peak = new_peak
                improved = True
            else:
                h[i] = old_val
        
        if not improved:
            # Try a few random restarts or larger moves
            # Try flipping pairs of positions
            for _ in range(100):
                if time.time() - start_time > max_time:
                    break
                i, j = np.random.choice(n, 2, replace=False)
                h[i] = -h[i]
                h[j] = -h[j]
                new_peak = _autocorr_peak(h)
                if new_peak < current_peak:
                    current_peak = new_peak
                    improved = True
                else:
                    h[i] = -h[i]
                    h[j] = -h[j]
            
            if not improved:
                break
    
    return h, current_peak


def construct():
    """Construct a signed step function with N=400 intervals that minimizes
    the autoconvolution upper bound."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Initialize with random ±1 sequence
    np.random.seed(42)  # Reproducible
    h = np.random.choice([-1.0, 1.0], size=N)
    
    # Run local search with time budget
    h, peak = _local_search(h, deadline - start_time - 5)
    
    # If time allows, try a few more restarts
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        # Try a couple more random initializations
        for seed in [123, 456]:
            if time.time() > deadline:
                break
            np.random.seed(seed)
            h_init = np.random.choice([-1.0, 1.0], size=N)
            h_candidate, peak_candidate = _local_search(h_init, deadline - time.time() - 5)
            if peak_candidate < peak:
                h = h_candidate
                peak = peak_candidate
    
    return h
# EVOLVE-BLOCK-END
