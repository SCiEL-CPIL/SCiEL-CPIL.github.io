import numpy as np
import time

N = 600

def compute_score(h):
    """Compute max_k S_k / S_0 where S_k is autocorrelation."""
    # Use FFT for fast autocorrelation
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-15):
        h = np.clip(h, 0, None)
    n_fft = 2 * N
    H = np.fft.rfft(h, n=n_fft)
    S = np.fft.irfft(H * np.conj(H), n=n_fft)[:N]
    S0 = np.dot(h, h)
    if S0 < 1e-30:
        return np.inf
    return np.max(S) / S0

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    # For nonnegative h, S_0 = sum(h_i^2) is always the max autocorrelation
    # by Cauchy-Schwarz, so normalized max is always 1.0.
    # Return a simple valid nonnegative step function.
    h = np.ones(N)
    return list(h)

construct()
