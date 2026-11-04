"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def _calc_max_autoconv(x):
    """
    Calculate the maximum absolute value of the aperiodic autocorrelation.
    x: numpy array of length N with values 1 or -1.
    """
    # Zero-pad to next power of 2 for efficient FFT
    # N=400, next power of 2 is 512? No, for autocorrelation we need size >= 2N-1.
    # 2*400 - 1 = 799. Next power of 2 is 1024.
    n_fft = 1024
    
    # Compute FFT
    X = np.fft.fft(x, n=n_fft)
    # Power spectrum
    P = X * np.conj(X)
    # Inverse FFT gives circular autocorrelation
    # r[k] = sum_{i=0}^{n-1} x[i] conj(x[(i+k) mod n])
    # For real x, r[k] = sum x[i] x[(i+k) mod n]
    # The first N-1 elements (indices 1 to N-1) correspond to positive lags A_k.
    r = np.fft.ifft(P).real
    
    # Extract lags 1 to N-1
    # r[0] is the energy N.
    # r[1:N] are A_1 to A_{N-1}
    # Note: Due to floating point errors, we might get tiny imaginary parts, 
    # but .real handles it.
    # We take absolute value and max
    return np.max(np.abs(r[1:N]))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    rng = np.random.RandomState(42)
    
    # Initialize with random ±1 values
    best_f = rng.choice([-1.0, 1.0], size=N)
    best_score = _calc_max_autoconv(best_f)
    
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
        
        new_f = current_f.copy()
        new_f[start:start + block_len] *= -1.0
        new_score = _calc_max_autoconv(new_f)
        
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
            score_new = _calc_max_autoconv(best_f)
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
            current_cost = _calc_max_autoconv(h_restart)
            
            # Quick greedy local search
            restart_deadline = min(deadline, time.time() + max(1.0, remaining_time / (num_restarts + 1)))
            for i in range(N):
                if time.time() > restart_deadline:
                    break
                old_val = h_restart[i]
                h_restart[i] = -old_val
                new_cost = _calc_max_autoconv(h_restart)
                if new_cost < current_cost:
                    current_cost = new_cost
                else:
                    h_restart[i] = old_val
            
            if current_cost < best_score:
                best_score = current_cost
                best_f = h_restart.copy()
    
    return best_f
# EVOLVE-BLOCK-END
