"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START


def _compute_c3(h):
    """
    Compute the autoconvolution upper bound C3 for a signed step function h.
    C3 is defined as the maximum absolute value of the autocorrelation at non-zero lags,
    normalized by the zero-lag value (which is N for binary +/-1 sequences).
    Lower is better.
    """
    # Compute autocorrelation via FFT
    # Pad to at least 2N to avoid circular convolution artifacts
    n_fft = 2 * N
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    r = np.fft.irfft(power, n=n_fft)
    
    # r[0] is lag 0 (sum of squares = N)
    # r[1] to r[N-1] are positive lags
    # r[n_fft - 1] ... r[n_fft - (N-1)] are negative lags
    # For real signals, autocorrelation is symmetric, so we only need positive lags
    
    # The zero-lag value
    r0 = r[0]
    
    # Max absolute value at non-zero lags
    # Positive lags: r[1:N]
    # Negative lags are symmetric, so we don't need to check them separately
    max_sidelobe = np.max(np.abs(r[1:N]))
    
    # C3 is the ratio (upper bound on normalized autocorrelation peak)
    return max_sidelobe / r0


def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    # Initial random sequence
    rng = np.random.default_rng(42)
    h = rng.choice([-1, 1], size=N).astype(np.float64)
    
    best_h = h.copy()
    best_c3 = _compute_c3(h)
    
    # Simulated annealing / local search
    # We'll use a simple hill-climbing with occasional random restarts
    
    step_count = 0
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() - start_time < deadline:
        # Check deadline frequently
        if time.time() - start_time > deadline - 5:
            break
        
        # Try flipping a random position
        idx = rng.integers(0, N)
        h[idx] *= -1
        
        new_c3 = _compute_c3(h)
        
        if new_c3 < best_c3:
            best_c3 = new_c3
            best_h = h.copy()
            no_improve_count = 0
        else:
            # Revert the flip
            h[idx] *= -1
            no_improve_count += 1
        
        step_count += 1
        
        # If we've had too many non-improving steps, try a random restart
        if no_improve_count > max_no_improve:
            # Random restart with some structure
            h = rng.choice([-1, 1], size=N).astype(np.float64)
            c3_new = _compute_c3(h)
            if c3_new < best_c3:
                best_c3 = c3_new
                best_h = h.copy()
            no_improve_count = 0
    
    return best_h
# EVOLVE-BLOCK-END
