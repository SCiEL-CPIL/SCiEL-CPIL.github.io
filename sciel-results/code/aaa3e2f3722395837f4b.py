"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _c3_bound(x):
    """Compute the autoconvolution upper bound C3 for a signed step function.
    
    The bound is related to the maximum of the autoconvolution of |x|,
    or more precisely, we want to minimize the L-infinity norm of the
    autoconvolution of the signed sequence.
    """
    # The autoconvolution upper bound C3 is typically the max of the
    # circular autocorrelation of the sequence
    # For a sequence x of length N, the autocorrelation at lag k is:
    # R(k) = sum_{i=0}^{N-k-1} x[i] * x[i+k]
    # The C3 bound is the maximum absolute value of the autocorrelation
    # over all non-zero lags, or sometimes the sum of squares of lags.
    
    # Let's compute the full autocorrelation
    # Using FFT for efficiency
    n = len(x)
    f = np.fft.fft(x, n=2*n)
    acf = np.fft.ifft(f * np.conj(f)).real[:n]
    
    # The C3 upper bound is typically max |acf[k]| for k != 0
    # Or it could be the L-infinity norm of the off-diagonal part
    # Let's use max absolute value of autocorrelation at non-zero lags
    max_lag = np.max(np.abs(acf[1:]))
    return max_lag


def _thue_morse(n):
    """Generate Thue-Morse sequence of length n as ±1 values."""
    # Thue-Morse: 1, -1, -1, 1, -1, 1, 1, -1, ...
    # It's defined by t(n) = popcount(n) mod 2, mapped to ±1
    tm = np.zeros(n, dtype=np.int8)
    for i in range(n):
        # Count number of 1-bits in i
        tm[i] = 1 if bin(i).count('1') % 2 == 0 else -1
    return tm


def _local_search(x, max_iters=5000, deadline=None):
    """Local search by flipping individual elements to reduce C3 bound."""
    x = x.copy()
    current_bound = _c3_bound(x)
    
    for i in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        best_gain = 0
        best_idx = -1
        
        # Try flipping each element
        for j in range(N):
            x[j] *= -1
            new_bound = _c3_bound(x)
            gain = current_bound - new_bound
            x[j] *= -1  # restore
            
            if gain > best_gain:
                best_gain = gain
                best_idx = j
        
        if best_idx >= 0:
            x[best_idx] *= -1
            current_bound -= best_gain
        else:
            break
    
    return x, current_bound


def construct():
    """Construct a signed step function with 400 equal intervals on [-1/4,1/4].
    
    Returns a sequence of 400 signed step heights in {-1, 1} that minimizes
    the autoconvolution upper bound C3.
    """
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Start with Thue-Morse sequence, which has known good autocorrelation properties
    x = _thue_morse(N)
    
    # Perform local search to improve
    x, bound = _local_search(x, max_iters=2000, deadline=deadline)
    
    # If we have time left, try more iterations or different starting points
    if time.time() < deadline - 10:
        # Try a few random restarts
        best_x = x
        best_bound = bound
        
        for seed in [1, 7, 13, 42, 99]:
            if time.time() > deadline - 5:
                break
            rng = np.random.default_rng(seed)
            x_rand = rng.choice([-1, 1], size=N)
            x_rand, b_rand = _local_search(x_rand, max_iters=1000, deadline=deadline)
            if b_rand < best_bound:
                best_x = x_rand
                best_bound = b_rand
        
        x = best_x
    
    return x
# EVOLVE-BLOCK-END
