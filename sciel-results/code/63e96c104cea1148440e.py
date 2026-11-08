import numpy as np
import time

N = 400
DEADLINE = 145.0  # Safety margin

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound for a signed step function.
    C3 = max_t |autoconv(t)| where autoconv is the convolution of f with itself.
    """
    M = 2 * N  # fine grid points
    dx = 0.5 / M
    
    # Vectorized construction of fine grid
    m_indices = np.arange(M)
    i_indices = (m_indices * N // M)
    i_indices = np.clip(i_indices, 0, N-1)
    f_fine = signed_step[i_indices]
    
    # Autoconvolution via FFT
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    c3 = np.max(np.abs(autoconv))
    
    return c3

def construct():
    start_time = time.time()
    
    # Initial best
    # Try all 1s
    step_ones = np.ones(N, dtype=np.int8)
    best_step = step_ones.copy()
    best_c3 = _compute_c3(best_step)
    
    # Try random initialization
    np.random.seed(42)
    for _ in range(5):
        rand_step = np.random.choice([-1, 1], size=N).astype(np.int8)
        c3_val = _compute_c3(rand_step)
        if c3_val < best_c3:
            best_c3 = c3_val
            best_step = rand_step.copy()
    
    # Simulated Annealing
    # We will run multiple restarts if time permits
    # Parameters
    T_start = 1.0  # Initial temperature
    T_end = 0.001  # Final temperature
    alpha = 0.999  # Cooling factor
    
    # Estimate number of iterations possible
    # One iteration: flip, compute c3 (FFT), compare.
    # Let's just run until deadline.
    
    current_step = best_step.copy()
    current_c3 = best_c3
    
    # Adaptive T or fixed schedule?
    # Fixed schedule is simpler.
    # We need to know how many steps we can take.
    # Let's assume 1000 steps/sec. 140 sec -> 140,000 steps.
    # Let's just use a time-based loop.
    
    T = T_start
    
    # To allow escaping local minima, we need T to be high enough.
    # The typical energy difference for a single flip is small.
    
    no_improve_count = 0
    patience = 5000  # If no improvement for 5000 steps, restart with new random state
    
    while time.time() - start_time < DEADLINE - 5.0:
        if no_improve_count >= patience:
            # Restart
            current_step = np.random.choice([-1, 1], size=N).astype(np.int8)
            current_c3 = _compute_c3(current_step)
            if current_c3 < best_c3:
                best_c3 = current_c3
                best_step = current_step.copy()
                no_improve_count = 0
            else:
                # If restart is worse, maybe revert to best? Or keep searching?
                # Let's revert to best to explore around it again with high T?
                # Or just keep the random one if it's not terrible?
                # Better: revert to best, but increase T temporarily?
                # Simpler: just break and do a final polish on best_step?
                # Let's try one more restart with best_step as seed.
                current_step = best_step.copy()
                current_c3 = best_c3
                T = T_start * 0.5 # Slightly lower temp for refinement?
                no_improve_count = 0
                continue

        # Pick random index
        idx = np.random.randint(N)
        old_val = current_step[idx]
        current_step[idx] = -old_val
        
        new_c3 = _compute_c3(current_step)
        delta = new_c3 - current_c3
        
        if delta < 0:
            # Accept
            current_c3 = new_c3
            if current_c3 < best_c3:
                best_c3 = current_c3
                best_step = current_step.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # Metropolis criterion
            # If T is very small, exp(-delta/T) is 0
            if T > 1e-9:
                # Optimization: if delta is large, probability is 0
                if delta < 10 * T: # Approximation to avoid underflow/calc if unlikely
                    prob = np.exp(-delta / T)
                    if np.random.random() < prob:
                        current_c3 = new_c3
                        if current_c3 < best_c3:
                            best_c3 = current_c3
                            best_step = current_step.copy()
                            no_improve_count = 0
                        else:
                            no_improve_count += 1
                    else:
                        # Revert
                        current_step[idx] = old_val
                        no_improve_count += 1
                else:
                    # Revert
                    current_step[idx] = old_val
                    no_improve_count += 1
            else:
                # Revert
                current_step[idx] = old_val
                no_improve_count += 1
        
        # Cool
        T *= alpha
        if T < T_end:
            T = T_end

    # Final Greedy Polish
    # Try to improve best_step with strict local search
    final_step = best_step.copy()
    final_c3 = best_c3
    improved = True
    while improved and time.time() - start_time < DEADLINE - 1.0:
        improved = False
        for i in range(N):
            old_val = final_step[i]
            final_step[i] = -old_val
            new_c3 = _compute_c3(final_step)
            if new_c3 < final_c3:
                final_c3 = new_c3
                improved = True
                if final_c3 < best_c3:
                    best_c3 = final_c3
                    best_step = final_step.copy()
            else:
                final_step[i] = old_val
                
    return best_step
