"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv_max_abs(f):
    """
    Compute the maximum absolute value of the aperiodic autocorrelation 
    excluding the zero-lag term.
    """
    n = len(f)
    # Zero-pad to 2*n to perform linear convolution via FFT
    size = 2 * n
    F = np.fft.fft(f, n=size)
    conv = np.fft.ifft(F * F).real[:n]
    # conv[0] is the sum of squares (self-correlation at lag 0)
    # We want max |conv[k]| for k > 0
    if n > 1:
        return np.max(np.abs(conv[1:]))
    return 0.0

def _block_flip(f, start, length):
    """
    Create a copy of f with a block of signs flipped.
    """
    f_new = f.copy()
    f_new[start:start + length] *= -1.0
    return f_new

def construct():
    start_time = time.time()
    # Safety margin: The evaluator kills at 150s. We stop at 140s.
    deadline = start_time + 140.0
    
    rng = np.random.RandomState(42)
    
    # Initialize with random ±1 values
    best_f = rng.choice([-1.0, 1.0], size=N)
    best_score = _autoconv_max_abs(best_f)
    
    # Phase 1: Simulated Annealing with Block Flips
    # Block flips allow escaping local minima more effectively than single flips.
    T = 10.0
    T_min = 0.01
    cooling = 0.998
    
    current_f = best_f.copy()
    current_score = best_score
    
    while time.time() < deadline - 30.0:
        # Choose a random block to flip
        # Shorter blocks for fine tuning, longer for coarse
        block_len = rng.randint(1, 15)
        start = rng.randint(0, N - block_len + 1)
        
        new_f = _block_flip(current_f, start, block_len)
        new_score = _autoconv_max_abs(new_f)
        
        delta = new_score - current_score
        # Accept if better, or with probability exp(-delta/T)
        if delta < 0 or (T > T_min and rng.rand() < np.exp(-delta / T)):
            current_f = new_f
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_f = new_f.copy()
        
        T *= cooling
        if T < T_min:
            T = T_min

    # Phase 2: Greedy Local Search with Single Flips
    # This refines the solution found by SA.
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        for i in range(N):
            old_val = best_f[i]
            best_f[i] = -old_val
            score_new = _autoconv_max_abs(best_f)
            if score_new < best_score:
                best_score = score_new
                improved = True
            else:
                best_f[i] = old_val
            if time.time() > deadline:
                return best_f
        if time.time() > deadline:
            return best_f

    # Phase 3: Multi-start with quick local search
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        num_restarts = min(10, int(remaining_time / 2.0))
        for _ in range(num_restarts):
            if time.time() >= deadline:
                break
            
            h_restart = rng.choice([-1.0, 1.0], size=N)
            current_cost = _autoconv_max_abs(h_restart)
            
            # Quick greedy local search
            restart_deadline = min(deadline, time.time() + max(1.0, remaining_time / (num_restarts + 1)))
            for i in range(N):
                if time.time() > restart_deadline:
                    break
                old_val = h_restart[i]
                h_restart[i] = -old_val
                new_cost = _autoconv_max_abs(h_restart)
                if new_cost < current_cost:
                    current_cost = new_cost
                else:
                    h_restart[i] = old_val
            
            if current_cost < best_score:
                best_score = current_cost
                best_f = h_restart.copy()

    return best_f
# EVOLVE-BLOCK-END
