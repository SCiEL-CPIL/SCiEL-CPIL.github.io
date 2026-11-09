"""Rewritten program: minimize peak sidelobe of autoconvolution for signed step function."""
import numpy as np
import time

N = 400

def _compute_autocorr(x):
    """Compute full autocorrelation (autoconvolution) of step function x."""
    # x is a sequence of N values (±1). The autoconvolution of the step function
    # on [-1/4, 1/4] with N equal intervals: the autoconvolution at lag k 
    # (for integer lags) is sum_{i} x[i] * x[i+k] for valid i.
    # For a step function, the continuous autoconvolution's peak sidelobe 
    # is related to the discrete autocorrelation sidelobes.
    # We compute the aperiodic autocorrelation: R[k] = sum_{i=0}^{N-1-k} x[i]*x[i+k] for k>=0
    # and by symmetry R[-k] = R[k].
    # The peak sidelobe is max(|R[k]|) for k != 0.
    # Using FFT for efficiency:
    n = len(x)
    size = 2 * n
    X = np.fft.rfft(x, size)
    acorr = np.fft.irfft(X * np.conj(X), size)[:n]
    # acorr[0] = sum x[i]^2 = N (for ±1)
    # Sidelobes: acorr[1:] and acorr[:0:-1] but for real, acorr is symmetric
    # Peak sidelobe = max(abs(acorr[1:]))
    return acorr

def _peak_sidelobe(x):
    """Return peak sidelobe magnitude of autoconvolution."""
    acorr = _compute_autocorr(x)
    return np.max(np.abs(acorr[1:]))

def _try_flip(x, i, best_val):
    """Try flipping position i, return (new_val, flipped_x) or (best_val, None)."""
    x[i] = -x[i]
    val = _peak_sidelobe(x)
    if val < best_val:
        return val, x.copy()
    x[i] = -x[i]
    return best_val, None

def construct():
    start_time = time.time()
    DEADLINE = 100  # safety margin before 120s limit
    
    # Initial sequence: use a Barker-like or m-sequence pattern
    # For N=400, use a quadratic residue sequence or just random with optimization
    # Start with a known good structure: alternating or based on Legendre symbol
    
    # Use a pseudo-random ±1 sequence as starting point
    np.random.seed(42)
    x = np.random.choice([-1.0, 1.0], size=N)
    
    best_val = _peak_sidelobe(x)
    best_x = x.copy()
    
    # Local search: iterate through positions, flip if it improves
    improved = True
    while improved:
        improved = False
        for i in range(N):
            if time.time() - start_time > DEADLINE:
                break
            new_val, new_x = _try_flip(best_x, i, best_val)
            if new_x is not None:
                best_val = new_val
                best_x = new_x
                improved = True
        if time.time() - start_time > DEADLINE:
            break
    
    # The score is the upper bound on C_3. The raw value (lower is better) is the peak sidelobe.
    # Return the sequence of signed step heights.
    return best_x

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
