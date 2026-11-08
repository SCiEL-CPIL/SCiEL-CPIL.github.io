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
    
    # Current state for SA
    current_step = best_step.copy()
    current_c3 = best_c3
    
    np.random.seed(42)
    
    # SA Parameters
    T0 = 0.5
    alpha = 0.999
    T = T0
    
    # Pre-compute some random indices or use on the fly? On the fly is fine.
    
    while True:
        # Check deadline with safety margin
        if time.time() - start_time > DEADLINE - 5.0:
            break
        
        # Randomly choose move type
        # As temperature drops, we might want to reduce multi-flips? 
        # Or keep it constant. Let's keep 50/50.
        if np.random.random() < 0.5:
            # Single flip
            idx = np.random.randint(N)
            old_val = current_step[idx]
            current_step[idx] = -old_val
            new_step = current_step
            # For reverting, we just flip back if rejected
            revert_func = lambda: setattr(current_step, '__setitem__', None) # dummy
            # Actually, easier to just copy or store old val
            # Let's just store the index and old val
            move_info = ('single', idx, old_val)
        else:
            # Multi-flip: flip 2-3 random distinct elements
            n_flips = np.random.randint(2, 4)
            idxs = np.random.choice(N, size=n_flips, replace=False)
            old_vals = [current_step[i] for i in idxs]
            for i in idxs:
                current_step[i] = -current_step[i]
            move_info = ('multi', idxs, old_vals)
            
        new_c3 = _compute_c3(current_step)
        
        # Acceptance Criteria
        delta = new_c3 - current_c3
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            # Accept
            current_c3 = new_c3
            if new_c3 < best_c3:
                best_c3 = new_c3
                best_step = current_step.copy()
        else:
            # Reject: revert
            if move_info[0] == 'single':
                current_step[move_info[1]] = move_info[2]
            else:
                for i, v in zip(move_info[1], move_info[2]):
                    current_step[i] = v
        
        # Cool down
        T *= alpha
        
        # Optional: If T is too low, we are in hill-climbing mode.
        # If we get stuck, we might want to reheat? 
        # For now, simple geometric cooling.
        
        # Safety check for T underflow
        if T < 1e-10:
             T = 1e-10

    return best_step
