import numpy as np
import time

N = 400
DEADLINE = 120.0

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound for a signed step function.
    The signed step function has N equal intervals on [-1/4, 1/4].
    C3 = max_t |autoconv(t)| where autoconv is the convolution of f with itself.
    """
    # signed_step is an array of N values in {-1, 1}
    # The function f(x) = signed_step[i] for x in [-1/4 + i/N, -1/4 + (i+1)/N)
    
    # Let h = 1/N be the interval width.
    # f(s) = signed_step[i] for s in [-1/4 + i*h, -1/4 + (i+1)*h)
    
    # Autoconvolution: (f*f)(t) = integral f(s) f(t-s) ds
    # For step functions, this can be computed efficiently.
    
    # Use a fine grid for numerical integration
    M = 4 * N  # fine grid resolution
    dx = (0.5) / M  # domain length is 0.5
    
    # Build f on fine grid
    f_fine = np.zeros(M)
    for i in range(N):
        start = int(i * M / N)
        end = int((i + 1) * M / N)
        f_fine[start:end] = signed_step[i]
    
    # Autoconvolution via FFT
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    # The autoconvolution is defined on [-0.5, 0.5]
    # C3 = max |autoconv|
    c3 = np.max(np.abs(autoconv))
    
    return c3

def construct():
    start_time = time.time()
    
    # Initialize with all ones (the seed)
    current_step = np.ones(N, dtype=np.int8)
    current_c3 = _compute_c3(current_step)
    
    # Track best solution
    best_step = current_step.copy()
    best_c3 = current_c3
    
    # Simulated annealing parameters
    np.random.seed(42)
    T_initial = 0.1
    T_final = 0.001
    cooling_rate = (T_final / T_initial) ** (1.0 / 100000)
    
    T = T_initial
    no_improve_count = 0
    patience = 20000
    
    while time.time() - start_time < DEADLINE - 2.0:
        # Try flipping a random element
        idx = np.random.randint(N)
        old_val = current_step[idx]
        current_step[idx] = -old_val
        
        new_c3 = _compute_c3(current_step)
        delta = new_c3 - current_c3
        
        # Accept if improvement or with probability exp(-delta/T) for uphill moves
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            current_c3 = new_c3
            if new_c3 < best_c3:
                best_c3 = new_c3
                best_step = current_step.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # Reject the move
            current_step[idx] = old_val
            no_improve_count += 1
        
        # Cool down
        T *= cooling_rate
        
        # If stuck too long, restart with a perturbation
        if no_improve_count >= patience:
            # Perturb the best solution
            current_step = best_step.copy()
            for _ in range(10):
                flip_idx = np.random.randint(N)
                current_step[flip_idx] = -current_step[flip_idx]
            current_c3 = _compute_c3(current_step)
            no_improve_count = 0
            T = T_initial * 0.5  # Reset temperature to medium
    
    return best_step
