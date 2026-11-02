"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def _autoconv_max(s):
    """Compute the maximum of the autoconvolution (circular) of the signed step function."""
    # Autoconvolution: c[k] = sum_i s[i] * s[(i+k) % N]
    # Using FFT for efficiency
    n_fft = 1 << (2 * N - 1).bit_length()
    f = np.fft.rfft(s, n=n_fft)
    c = np.fft.irfft(f * np.conj(f), n=n_fft)
    # The autoconvolution values are in c[0:N] (and symmetric part)
    # We want the maximum over all lags
    return np.max(c[:N])

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    # Initialize with random signs
    rng = np.random.default_rng(42)
    best_s = rng.choice([-1.0, 1.0], size=N)
    best_score = _autoconv_max(best_s)
    
    # Simulated annealing with block-flip moves
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.999
    
    # Block sizes to try
    block_sizes = [1, 2, 4, 8, 16, 32]
    
    iters = 0
    while T > T_min and time.time() < deadline:
        iters += 1
        # Choose a random block size (larger blocks with lower probability)
        bs = rng.choice(block_sizes, p=[0.2, 0.2, 0.2, 0.15, 0.15, 0.1])
        start = rng.integers(0, N)
        
        # Create a copy for the proposed move
        s_new = best_s.copy()
        # Flip signs in the block (circular)
        for i in range(start, min(start + bs, N)):
            s_new[i] *= -1
        if start + bs > N:
            for i in range(0, start + bs - N):
                s_new[i] *= -1
        
        # Compute new score
        new_score = _autoconv_max(s_new)
        delta = new_score - best_score
        
        # Accept if better or with probability exp(-delta/T)
        if delta < 0 or rng.random() < np.exp(-delta / T):
            best_s = s_new
            best_score = new_score
        
        T *= cooling_rate
    
    # Final polish: local search with single-bit flips
    if time.time() < deadline - 5.0:
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for i in range(N):
                s_trial = best_s.copy()
                s_trial[i] *= -1
                trial_score = _autoconv_max(s_trial)
                if trial_score < best_score:
                    best_s = s_trial
                    best_score = trial_score
                    improved = True
    
    return best_s
# EVOLVE-BLOCK-END
