"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400
W = 1.0 / N  # interval width

def compute_autoconv(h):
    """Compute autoconvolution of step function using FFT."""
    # Pad to avoid circular convolution
    size = 2 * N
    f = np.zeros(size)
    f[:N] = h
    # Autoconvolution via FFT
    F = np.fft.fft(f, size)
    ac = np.fft.ifft(F * np.conj(F), size).real
    # The autoconvolution at lag k (for k=0 to N-1) is ac[k] * W
    # For negative lags, ac[size-k] = ac[k] by symmetry
    # Return the autoconvolution values at discrete lags 0 to N-1
    return ac[:N] * W

def objective(h):
    """Compute the upper bound on C3 - using max absolute autoconvolution."""
    ac = compute_autoconv(h)
    # The upper bound might be the max absolute value
    # For signed functions, we want to minimize this
    return np.max(np.abs(ac))

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initial configuration: alternating pattern to reduce autoconvolution
    h = np.ones(N)
    h[::2] = -1
    best_h = h.copy()
    best_val = objective(h)
    
    # Simulated annealing
    T = 1.0
    T_min = 1e-4
    cooling = 0.9995
    step_count = 0
    
    while T > T_min:
        if time.time() > deadline:
            break
        
        # Random flip
        idx = np.random.randint(N)
        h_new = h.copy()
        h_new[idx] = -h_new[idx]
        
        val_new = objective(h_new)
        delta = val_new - best_val
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            h = h_new
            if val_new < best_val:
                best_val = val_new
                best_h = h_new.copy()
        
        T *= cooling
        step_count += 1
    
    # Final polish: local search with single flips
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for idx in range(N):
                if time.time() > deadline - 2:
                    break
                h_trial = best_h.copy()
                h_trial[idx] = -h_trial[idx]
                val_trial = objective(h_trial)
                if val_trial < best_val - 1e-10:
                    best_h = h_trial
                    best_val = val_trial
                    improved = True
    
    return best_h

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
