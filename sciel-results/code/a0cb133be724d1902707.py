"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_c3(h):
    """Compute the upper bound C3 from the signed step heights.
    C3 is the L2 norm of the autoconvolution (linear convolution of h with itself).
    """
    # Linear convolution of h with itself
    conv = np.convolve(h, h, mode='full')
    return np.sqrt(np.sum(conv ** 2))

def _compute_c3_fast(h):
    """Faster computation using FFT for the autoconvolution."""
    n = len(h)
    # Size for FFT to avoid circular convolution
    nfft = 1
    while nfft < 2 * n - 1:
        nfft *= 2
    H = np.fft.rfft(h, n=nfft)
    conv = np.fft.irfft(H * H, n=nfft)[:2 * n - 1]
    return np.sqrt(np.sum(conv ** 2))

def _random_flip_indices(h, num_flips):
    """Return indices for multi-flip move."""
    idx = np.random.choice(len(h), size=num_flips, replace=False)
    return idx

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Initialize with random signed step function
    best_h = np.random.choice([-1.0, 1.0], size=N)
    best_c3 = _compute_c3_fast(best_h)
    
    # Local search with multi-flip moves
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        # Choose number of flips (2-3 for escaping local minima)
        num_flips = np.random.choice([2, 3], p=[0.7, 0.3])
        
        # Generate candidate
        candidate = best_h.copy()
        idx = np.random.choice(N, size=num_flips, replace=False)
        candidate[idx] *= -1  # Flip signs
        
        new_c3 = _compute_c3_fast(candidate)
        
        if new_c3 < best_c3:
            best_h = candidate
            best_c3 = new_c3
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if no_improve_count > max_no_improve:
            # Restart with new random initialization
            best_h = np.random.choice([-1.0, 1.0], size=N)
            best_c3 = _compute_c3_fast(best_h)
            no_improve_count = 0
            max_no_improve = 500 + np.random.randint(0, 500)
    
    return best_h
# EVOLVE-BLOCK-END

