N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autocorr_peak(h):
    """Compute max |autocorrelation| at non-zero lags."""
    n = len(h)
    if n <= 1:
        return 0.0
    # Use FFT for efficient autocorrelation
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    # acf[0] is lag 0, we want max |acf[k]| for k=1..n-1
    return np.max(np.abs(acf[1:n]))


def _compute_peak_fast(h, fft_size):
    """Fast peak computation using precomputed FFT size."""
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    return np.max(np.abs(acf[1:N]))


def _local_search_full(h, start_time, time_limit, fft_size):
    """
    Greedy local search on the full ±1 vector.
    Iterates through positions, flipping if it improves the peak.
    """
    current_peak = _compute_peak_fast(h, fft_size)
    best_peak = current_peak
    best_h = h.copy()
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(N):
            if (time.time() - start_time) > time_limit:
                break
            # Try flipping position i
            h[i] = -h[i]
            new_peak = _compute_peak_fast(h, fft_size)
            if new_peak < best_peak - 1e-12:
                best_peak = new_peak
                best_h = h.copy()
                improved = True
            else:
                h[i] = -h[i]  # revert
    
    return best_h, best_peak


def _sparse_local_search(h, start_time, time_limit, fft_size):
    """
    Local search that also allows changing sparsity (setting values to 0).
    This is a structural change from pure ±1 to allow zeros.
    """
    current_peak = _compute_peak_fast(h, fft_size)
    best_peak = current_peak
    best_h = h.copy()
    
    # Try to set some positions to 0
    for _ in range(50):
        if (time.time() - start_time) > time_limit:
            break
        
        # Randomly pick a position to try setting to 0
        i = np.random.randint(N)
        if h[i] == 0:
            continue
            
        old_val = h[i]
        h[i] = 0.0
        new_peak = _compute_peak_fast(h, fft_size)
        
        if new_peak < best_peak - 1e-12:
            best_peak = new_peak
            best_h = h.copy()
        else:
            h[i] = old_val
    
    # Also try flipping signs of non-zero positions
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(N):
            if (time.time() - start_time) > time_limit:
                break
            if h[i] == 0:
                continue
            # Try flipping position i
            h[i] = -h[i]
            new_peak = _compute_peak_fast(h, fft_size)
            if new_peak < best_peak - 1e-12:
                best_peak = new_peak
                best_h = h.copy()
                improved = True
            else:
                h[i] = -h[i]  # revert
    
    return best_h, best_peak


def construct():
    """
    Construct a signed step function with N=400 intervals that minimizes
    the autoconvolution upper bound (max absolute autocorrelation at non-zero lags).
    
    Strategy: 
    1. Initialize with random ±1 pattern
    2. Use greedy local search to flip positions that reduce the peak
    3. Also explore sparse solutions (with zeros) via a separate local search
    4. Time-budgeted to finish well within 120s
    """
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    # Precompute FFT size
    fft_size = 1
    while fft_size < 2 * N:
        fft_size *= 2
    
    # Initialize with random ±1 pattern
    np.random.seed(42)
    h_full = np.random.choice([-1.0, 1.0], size=N)
    
    # Phase 1: Full ±1 local search
    h_best_full, peak_full = _local_search_full(h_full, start_time, time_limit * 0.4, fft_size)
    
    # Phase 2: Sparse local search (allow zeros)
    # Start from the best full solution and try to make it sparse
    h_sparse = h_best_full.copy()
    h_best_sparse, peak_sparse = _sparse_local_search(h_sparse, start_time, time_limit * 0.8, fft_size)
    
    # Choose the better of the two
    if peak_sparse < peak_full:
        return h_best_sparse
    else:
        return h_best_full
# EVOLVE-BLOCK-END
