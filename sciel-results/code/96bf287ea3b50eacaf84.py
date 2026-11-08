import numpy as np
import time

N = 400
DEADLINE = 120.0

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound for a signed step function.
    """
    M = 4 * N
    dx = (0.5) / M
    
    f_fine = np.zeros(M)
    for i in range(N):
        start = int(i * M / N)
        end = int((i + 1) * M / N)
        f_fine[start:end] = signed_step[i]
    
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
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
    
    np.random.seed(42)
    
    # Simulated annealing parameters
    T_initial = 0.1
    T_final = 0.001
    cooling_rate = 0.999
    
    T = T_initial
    
    # Run simulated annealing until deadline
    while time.time() - start_time < DEADLINE - 5.0:
        idx = np.random.randint(N)
        old_val = current_step[idx]
        current_step[idx] = -old_val
        
        new_c3 = _compute_c3(current_step)
        
        delta_c3 = new_c3 - current_c3
        
        # Accept if better, or with probability exp(-delta/T) if worse
        if delta_c3 < 0 or np.random.random() < np.exp(-delta_c3 / T):
            current_step[idx] = current_step[idx]  # Keep the flip
            current_c3 = new_c3
            
            if new_c3 < best_c3:
                best_c3 = new_c3
                best_step = current_step.copy()
        else:
            # Revert
            current_step[idx] = old_val
        
        # Cool down
        T *= cooling_rate
        if T < T_final:
            T = T_final
    
    return best_step
