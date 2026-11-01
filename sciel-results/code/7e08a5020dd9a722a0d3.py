"""
Search for a signed step function with 400 intervals on [-1/4, 1/4]
that minimizes the peak of the autoconvolution (C3 upper bound).

The seed returns all-ones, which has maximal constructive interference at lag 0.
We use local search with random restarts and a deadline guard to stay within 120s.
"""
import numpy as np
import time

N = 400
SEED = 42
DEADLINE_SECONDS = 115  # safety margin below 120s

def _autoconv_peak(x):
    """
    Compute the peak of the circular autoconvolution of x.
    For a step function on [-1/4, 1/4] with N equal intervals, the autoconvolution
    at lag k is sum_i x[i] * x[i-k mod N]. The peak is the max absolute value.
    Actually, for the C3 bound, we want the max of the autoconvolution values.
    Let's compute the full circular convolution and take the max.
    """
    # Use FFT for efficient circular convolution
    nfft = 1 << (2 * N - 1).bit_length()  # next power of 2 >= 2N-1
    X = np.fft.rfft(x, n=nfft)
    conv = np.fft.irfft(X * X, n=nfft)[:2 * N - 1]
    # The autoconvolution has symmetry; the peak is at the center or we take max
    return np.max(np.abs(conv))

def _autoconv_peak_direct(x):
    """Direct computation for verification and small cases."""
    # Circular convolution
    conv = np.zeros(2 * N - 1)
    for k in range(2 * N - 1):
        if k < N:
            # sum_i x[i] * x[i + k] for i where both are in range
            # Actually circular: x[i] * x[(i+k) mod N]
            # Let's just use np.correlate or manual
            pass
    # Simpler: use numpy's correlate with 'full' mode then take the relevant part
    # For circular autoconvolution, we can pad and use FFT
    # Let's stick with FFT version
    nfft = 1 << (2 * N - 1).bit_length()
    X = np.fft.rfft(x, n=nfft)
    conv = np.fft.irfft(X * X, n=nfft)[:2 * N - 1]
    return np.max(np.abs(conv))

def _flip_bit(x, idx):
    """Flip the sign at position idx (x[i] = -x[i])."""
    x = x.copy()
    x[idx] = -x[idx]
    return x

def _random_signed_array(rng):
    """Generate a random array of +1/-1."""
    return rng.choice([-1, 1], size=N)

def _local_search(x, rng, max_iter, deadline):
    """
    Local search: try flipping individual bits to reduce the peak.
    Returns the best x found and its peak value.
    """
    current_peak = _autoconv_peak(x)
    best_x = x.copy()
    best_peak = current_peak
    current_x = x.copy()
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Try flipping each bit and see if it improves
        best_improvement = 0
        best_idx = -1
        best_new_peak = current_peak
        
        for i in range(N):
            # Flip bit i
            current_x[i] = -current_x[i]
            new_peak = _autoconv_peak(current_x)
            current_x[i] = -current_x[i]  # unflip
            
            improvement = current_peak - new_peak
            if improvement > best_improvement:
                best_improvement = improvement
                best_idx = i
                best_new_peak = new_peak
        
        if best_idx >= 0:
            current_x[best_idx] = -current_x[best_idx]
            current_peak = best_new_peak
            if current_peak < best_peak:
                best_peak = current_peak
                best_x = current_x.copy()
        else:
            # No improvement found, break
            break
    
    return best_x, best_peak

def construct():
    rng = np.random.default_rng(SEED)
    deadline = time.time() + DEADLINE_SECONDS
    
    # Start with all-ones (the seed)
    x = np.ones(N, dtype=np.float64)
    best_x = x.copy()
    best_peak = _autoconv_peak(x)
    
    # Also try a few random starts
    for start_iter in range(20):
        if time.time() > deadline - 5:
            break
        
        # Random signed array
        x_rand = _random_signed_array(rng)
        
        # Local search from this start
        x_best, peak_best = _local_search(x_rand, rng, max_iter=100, deadline=deadline)
        
        if peak_best < best_peak:
            best_peak = peak_best
            best_x = x_best.copy()
    
    # Also do local search from the all-ones start
    if time.time() < deadline - 5:
        x_best, peak_best = _local_search(np.ones(N, dtype=np.float64), rng, max_iter=200, deadline=deadline)
        if peak_best < best_peak:
            best_peak = peak_best
            best_x = x_best.copy()
    
    # Return the sequence of signed step heights
    return best_x

# For testing
if __name__ == "__main__":
    x = construct()
    peak = _autoconv_peak(x)
    print(f"Peak autoconvolution: {peak}")
    print(f"Number of +1s: {np.sum(x > 0)}, Number of -1s: {np.sum(x < 0)}")
