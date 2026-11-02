"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_bound(x):
    """Compute the autoconvolution upper bound (max off-peak autocorrelation magnitude)."""
    # Compute autocorrelation via FFT
    n = len(x)
    f = np.fft.fft(x, n=2*n)
    acf = np.fft.ifft(f * np.conj(f)).real[:n]
    # Off-peak: exclude lag 0
    return np.max(np.abs(acf[1:]))


def _try_flip(x, i, current_bound):
    """Try flipping sign at position i, return new bound if improved."""
    x[i] = -x[i]
    new_bound = _compute_bound(x)
    x[i] = -x[i]
    return new_bound


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Initialize with alternating pattern (good starting point for low autocorrelation)
    x = np.ones(N)
    x[1::2] = -1
    
    best_x = x.copy()
    best_bound = _compute_bound(best_x)
    
    # Metropolis-Hastings with adaptive temperature
    temperature = 1.0
    current_bound = best_bound
    current_x = best_x.copy()
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        # Try a batch of flips
        for _ in range(100):
            if time.time() >= deadline:
                break
            
            i = rng.integers(0, N)
            old_val = current_x[i]
            current_x[i] = -old_val
            
            new_bound = _compute_bound(current_x)
            
            # Metropolis acceptance criterion
            if new_bound < current_bound:
                # Always accept improvement
                current_bound = new_bound
                if new_bound < best_bound:
                    best_bound = new_bound
                    best_x = current_x.copy()
            else:
                # Accept with probability exp(-(new_bound - current_bound) / temperature)
                delta = new_bound - current_bound
                if delta > 0:
                    prob = np.exp(-delta / temperature)
                    if rng.random() > prob:
                        # Reject: revert
                        current_x[i] = old_val
                # If delta <= 0, already accepted above
            
            # Gradually cool temperature
            temperature *= 0.9999
        
        # Periodic restart from best if stuck
        if current_bound > best_bound * 1.01:
            current_x = best_x.copy()
            current_bound = best_bound
    
    return best_x
# EVOLVE-BLOCK-END

