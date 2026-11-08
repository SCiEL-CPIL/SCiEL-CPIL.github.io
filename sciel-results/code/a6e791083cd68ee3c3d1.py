import numpy as np
import time
import math

N = 400
DEADLINE = 120.0

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound for a signed step function.
    C3 = max_t |autoconv(t)| where autoconv is the convolution of f with itself.
    """
    M = 4 * N
    dx = 0.5 / M
    
    f_fine = np.repeat(signed_step, M // N)
    
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    c3 = np.max(np.abs(autoconv))
    return c3

def construct():
    start_time = time.time()
    
    np.random.seed(42)
    
    # Simulated annealing parameters
    T_initial = 0.01
    T_final = 0.0001
    cooling_rate = 0.999
    
    # Initialize with all ones (the seed)
    current_step = np.ones(N, dtype=np.int8)
    current_c3 = _compute_c3(current_step)
    
    best_step = current_step.copy()
    best_c3 = current_c3
    
    T = T_initial
    
    while True:
        # Check deadline with safety margin
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 2.0:
            break
        
        # Try flipping a random element
        idx = np.random.randint(N)
        current_step[idx] = -current_step[idx]
        
        new_c3 = _compute_c3(current_step)
        delta_c3 = new_c3 - current_c3
        
        # Accept if better, or with probability exp(-delta/T) if worse
        if delta_c3 < 0:
            # Always accept improvements
            current_c3 = new_c3
            if new_c3 < best_c3:
                best_c3 = new_c3
                best_step = current_step.copy()
        else:
            # Probabilistic acceptance for uphill moves
            if T > 1e-10:
                accept_prob = math.exp(-delta_c3 / T)
                if np.random.random() < accept_prob:
                    current_c3 = new_c3
                else:
                    # Reject: revert the flip
                    current_step[idx] = -current_step[idx]
            else:
                # At very low temperature, only accept improvements
                current_step[idx] = -current_step[idx]
        
        # Cool down
        T *= cooling_rate
    
    return best_step
