"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400
W = 1.0 / N  # width of each interval = 1/800... wait, [-1/4, 1/4] has length 1/2
# Actually [-1/4, 1/4] has length 1/2, so each interval has width (1/2)/400 = 1/800
W = 0.5 / N

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconvolution_max_abs(signs):
    """Compute the L-infinity norm of the autoconvolution of the signed step function."""
    # signs: array of N values in {+1, -1}
    # The autoconvolution at lag k (in interval units) can be computed via FFT
    # Pad to length 2N-1 for linear convolution
    n = len(signs)
    fft_size = 1
    while fft_size < 2 * n - 1:
        fft_size *= 2
    
    # Use FFT to compute linear convolution of signs with itself
    f = np.zeros(fft_size)
    f[:n] = signs
    F = np.fft.fft(f)
    conv = np.fft.ifft(F * F).real[:2*n-1]
    
    # Scale by interval width W
    conv_scaled = conv * W
    
    return np.max(np.abs(conv_scaled))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initialize with random signs
    best_signs = np.random.choice([1, -1], size=N)
    best_score = _autoconvolution_max_abs(best_signs)
    
    # Local search: flip individual signs to reduce max abs autoconvolution
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            # Try flipping sign i
            old_val = best_signs[i]
            best_signs[i] = -old_val
            new_score = _autoconvolution_max_abs(best_signs)
            if new_score < best_score:
                best_score = new_score
                improved = True
            else:
                best_signs[i] = old_val
    
    # Return the best signs found
    return best_signs
# EVOLVE-BLOCK-END
