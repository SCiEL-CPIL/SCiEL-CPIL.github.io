"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconvolution_max_abs(f):
    """Compute max |f*f| over all lags using FFT."""
    n = len(f)
    conv = np.fft.ifft(np.fft.fft(f, 2*n) ** 2).real
    return np.max(np.abs(conv))


def construct():
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin before 120s
    
    # Start with alternating ±1
    f = np.ones(N)
    f[::2] = -1
    
    best_f = f.copy()
    best_val = _autoconvolution_max_abs(f)
    
    # Local search: try flipping each position, keep if improves
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            f[i] *= -1
            val = _autoconvolution_max_abs(f)
            if val < best_val - 1e-12:
                best_val = val
                best_f = f.copy()
                improved = True
            else:
                f[i] *= -1  # revert
    
    return best_f
# EVOLVE-BLOCK-END
