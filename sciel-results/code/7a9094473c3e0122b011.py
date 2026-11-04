"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconv_max_abs(f):
    """Compute max absolute value of aperiodic autocorrelation (excluding lag 0)."""
    n = len(f)
    size = 2 * n
    F = np.fft.fft(f, n=size)
    conv = np.fft.ifft(F * F).real[:n]
    # conv[0] is the self-convolution at lag 0
    return np.max(np.abs(conv[1:]))


def _block_flip(f, start, length):
    """Create a new array with a block of signs flipped."""
    f_new = f.copy()
    f_new[start:start + length] *= -1.0
    return f_new


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    rng = np.random.RandomState(42)
    
    # Initialize with random ±1 values
    best_f = rng.choice([-1.0, 1.0], size=N)
    best_score = _autoconv_max_abs(best_f)
    
    # Phase 1: Simulated annealing with block flips
    T = 20.0
    T_min = 0.01
    cooling = 0.997
    current_f = best_f.copy()
    current_score = best_score
    
    while time.time() < deadline - 40.0:
        # Choose a random block to flip
        block_len = rng.randint(1, 20)  # Block length 1 to 20
        start = rng.randint(0, N - block_len + 1)
        
        new_f = _block_flip(current_f, start, block_len)
        new_score = _autoconv_max_abs(new_f)
        
        delta = new_score - current_score
        if delta < 0 or (T > T_min and rng.rand() < np.exp(-delta / T)):
            current_f = new_f
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_f = new_f.copy()
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    # Phase 2: Greedy local search with single flips
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
        num_restarts = min(20, int(remaining_time / 3.0))
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
