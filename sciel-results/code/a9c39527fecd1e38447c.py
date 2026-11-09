"""Program for optimizing a signed step function to minimize autoconvolution upper bound."""
import numpy as np
import time

N = 400
DEADLINE = 115  # seconds, safety margin under 120s


def compute_metric(h):
    """
    Compute the upper bound on C3 for a signed step function.
    h: array of length N with values +/-1.
    Returns a scalar (lower is better).
    """
    # Autocorrelation using FFT
    n_fft = 1 << (2 * N).bit_length()
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    r = np.fft.irfft(power, n=n_fft)
    
    # r has length n_fft. The linear autocorrelation is embedded:
    # r[0] = sum h[i]^2 = N (lag 0)
    # r[1..N-1] = autocorrelation at lags 1..N-1
    # r[n_fft-N+1..n_fft-1] = autocorrelation at lags -(N-1)..-1
    # For real sequences, r[k] = r[n_fft-k]
    
    # Sidelobes: all lags except 0
    # Normalized autocorrelation
    r_norm = r / N
    
    # Peak sidelobe magnitude
    # Exclude lag 0 (index 0) and the wrap-around equivalent
    sidelobes = np.concatenate([r_norm[1:N], r_norm[n_fft-N+1:n_fft]])
    return np.max(np.abs(sidelobes))


def construct():
    """Construct a signed step function to minimize the autoconvolution upper bound."""
    start_time = time.time()
    
    # Start with random initialization
    rng = np.random.default_rng(42)
    best_h = rng.choice([-1, 1], size=N)
    best_score = compute_metric(best_h)
    
    # Local search: flip individual bits
    improved = True
    while improved:
        improved = False
        for i in range(N):
            if time.time() - start_time > DEADLINE:
                break
            # Try flipping bit i
            old_val = best_h[i]
            best_h[i] = -old_val
            new_score = compute_metric(best_h)
            if new_score < best_score:
                best_score = new_score
                improved = True
            else:
                best_h[i] = old_val
        if time.time() - start_time > DEADLINE:
            break
    
    return best_h
