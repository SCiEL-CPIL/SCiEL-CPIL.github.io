"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def compute_score(x):
    """Compute max|autoconv| / (sum x)^2 for sequence x using FFT for speed."""
    # Pad to next power of 2 or at least 2*N for linear convolution via FFT
    size = 2 * N
    # FFT autoconvolution
    f = np.fft.rfft(x, n=size)
    autoconv = np.fft.irfft(f * np.conj(f), n=size)
    # The linear autoconvolution is in the center part? 
    # np.correlate(x, x, 'full') has length 2*N-1.
    # FFT convolution gives circular conv. With padding 2*N, it matches linear conv.
    # The peak is the max absolute value of the full autoconvolution.
    max_abs_ac = np.max(np.abs(autoconv))
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return float('inf')
    return max_abs_ac / (integral * integral)


def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit (evaluator kills at 150s, but task says 120s finish)
    
    # Start with all-ones
    x = np.ones(N, dtype=np.float64)
    best_x = x.copy()
    best_score = compute_score(best_x)
    
    # Local search: flip individual bits
    # We use a simple random restart local search with FFT-based scoring
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        # Perform a batch of flips
        for _ in range(1000):
            if time.time() > deadline:
                break
            idx = rng.integers(N)
            x[idx] = -x[idx]
            new_score = compute_score(x)
            
            if new_score < best_score:
                best_score = new_score
                best_x = x.copy()
            else:
                x[idx] = -x[idx]
                
    return best_x
# EVOLVE-BLOCK-END
