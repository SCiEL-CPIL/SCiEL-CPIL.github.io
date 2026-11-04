"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n (0/1)."""
    tm = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        tm[i] = tm[i >> 1] ^ (i & 1)
    return tm

def _linear_autocorr(x):
    """Compute linear autocorrelation of x (real) using FFT."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size <<= 1
    F = np.fft.fft(x, size)
    acf = np.fft.ifft(F * np.conj(F)).real
    return acf[:n]  # lags 0..n-1

def _objective(x):
    """Return the max |autocorrelation| for lags 1..N-1."""
    acf = _linear_autocorr(x)
    return np.max(np.abs(acf[1:]))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Initialize with Thue-Morse (map 0->1, 1->-1)
    tm = _thue_morse(N)
    x = 1.0 - 2.0 * tm.astype(np.float64)  # 0->1, 1->-1
    
    best_x = x.copy()
    best_obj = _objective(best_x)
    current_obj = best_obj
    
    # Local search with simulated annealing
    T = 0.1  # initial temperature (small since we want to minimize)
    cooling = 0.99995
    min_T = 1e-10
    
    while time.time() < deadline:
        T *= cooling
        if T < min_T:
            T = min_T
        
        # Try a batch of random flips
        batch_size = 100
        for _ in range(batch_size):
            if time.time() >= deadline:
                break
            
            i = np.random.randint(0, N)
            old_val = x[i]
            x[i] = -old_val
            new_obj = _objective(x)
            
            delta = new_obj - current_obj
            
            if delta < 0 or (T > min_T and np.random.random() < np.exp(-delta / T)):
                current_obj = new_obj
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_x = x.copy()
            else:
                x[i] = old_val  # revert
    
    return best_x
# EVOLVE-BLOCK-END

