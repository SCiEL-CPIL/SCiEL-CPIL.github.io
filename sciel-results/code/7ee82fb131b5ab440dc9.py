"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv_peak(h):
    """Compute max absolute value of linear autoconvolution of step function h."""
    N = len(h)
    # Linear autoconvolution: for shift m, sum_{i=0}^{N-1-m} h[i]*h[i+m]
    # We want max over m of |conv[m]|
    # Use FFT for efficiency
    # Zero-pad to avoid circular convolution issues
    size = 2 * N
    f = np.zeros(size)
    f[:N] = h
    # Autoconvolution via FFT
    F = np.fft.fft(f)
    conv = np.fft.ifft(F * F).real[:N]
    return np.max(np.abs(conv))

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    # Initialize with random signs
    h = np.random.choice([-1, 1], size=N)
    best_h = h.copy()
    best_score = _autoconv_peak(h)
    
    # Local search with simulated annealing
    current_score = best_score
    T = 1.0  # initial temperature
    cooling = 0.999
    
    while time.time() < deadline:
        # Pick a random index to flip
        idx = np.random.randint(N)
        h[idx] *= -1
        
        new_score = _autoconv_peak(h)
        
        if new_score < current_score or np.random.random() < np.exp(-(new_score - current_score) / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = h.copy()
        
        T *= cooling
        if T < 0.01:
            T = 0.01
    
    return best_h
# EVOLVE-BLOCK-END
