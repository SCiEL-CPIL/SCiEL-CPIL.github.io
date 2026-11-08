import numpy as np
import time

N = 400
DEADLINE = 120.0
M = 4 * N  # fine grid size
dx = 0.5 / M  # domain length is 0.5

def _compute_c3_fast(signed_step):
    """
    Fast computation of C3 = max_t |autoconv(t)| for signed step function.
    Uses vectorized fine-grid construction and FFT.
    """
    # Vectorized construction of fine grid from step function
    # Each step interval i maps to M//N fine grid points
    f_fine = np.repeat(signed_step.astype(np.float64), M // N)
    
    # Autoconvolution via FFT
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    return np.max(np.abs(autoconv))

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound for a signed step function.
    C3 = max_t |autoconv(t)| where autoconv is the convolution of f with itself.
    """
    return _compute_c3_fast(signed_step)

def construct():
    start_time = time.time()
    
    # Initialize with all ones (the seed)
    best_step = np.ones(N, dtype=np.int8)
    best_c3 = _compute_c3(best_step)
    
    np.random.seed(42)
    
    no_improve_count = 0
    patience = 8000
    
    while True:
        # Check deadline with safety margin
        if time.time() - start_time > DEADLINE - 5.0:
            break
        
        # Randomly choose move type: single flip (70%) or multi-flip (30%)
        if np.random.random() < 0.7:
            # Single flip
            idx = np.random.randint(N)
            old_vals = [best_step[idx]]
            best_step[idx] = -best_step[idx]
        else:
            # Multi-flip: flip 2-3 random distinct elements
            n_flips = np.random.randint(2, 4)
            idxs = np.random.choice(N, size=n_flips, replace=False)
            old_vals = [best_step[i] for i in idxs]
            for i in idxs:
                best_step[i] = -best_step[i]
        
        new_c3 = _compute_c3(best_step)
        
        if new_c3 < best_c3:
            best_c3 = new_c3
            no_improve_count = 0
        else:
            # Revert
            if len(old_vals) == 1:
                best_step[old_vals[0]] = old_vals[0]
            else:
                for i, v in zip(idxs, old_vals):
                    best_step[i] = v
            no_improve_count += 1
        
        if no_improve_count >= patience:
            break
    
    return best_step
