"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    """Construct a signed step function to minimize the autoconvolution upper bound C3."""
    deadline = time.time() + 120  # safety margin before 150s limit
    
    # Start with a random +/-1 sequence
    best_seq = None
    best_score = float('inf')
    
    # Generate several random sequences and pick the best
    rng = np.random.default_rng(42)
    
    for i in range(1000):
        if time.time() > deadline - 10:
            break
        
        seq = rng.choice([-1.0, 1.0], size=N)
        
        # Compute max off-peak autocorrelation
        # Use FFT for efficiency
        n_fft = 1
        while n_fft < 2 * N:
            n_fft *= 2
        
        f = np.fft.fft(seq, n=n_fft)
        acf = np.fft.ifft(f * np.conj(f)).real[:N]
        
        # Off-peak: lags 1 to N-1
        if N > 1:
            off_peak_max = np.max(np.abs(acf[1:]))
        else:
            off_peak_max = 0.0
        
        if off_peak_max < best_score:
            best_score = off_peak_max
            best_seq = seq.copy()
    
    if best_seq is None:
        best_seq = rng.choice([-1.0, 1.0], size=N)
    
    return best_seq
# EVOLVE-BLOCK-END
