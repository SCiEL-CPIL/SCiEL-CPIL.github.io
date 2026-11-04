"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n as +1/-1."""
    tm = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        tm[i] = tm[i >> 1] ^ (i & 1)
    return 1 - 2 * tm  # convert 0/1 to +1/-1

def _autoconv_peak(x):
    """Compute the peak of the autoconvolution (sum of |autocorr[k]|)."""
    # Autocorrelation via FFT
    n = len(x)
    f = np.fft.fft(x, n=2*n)
    ac = np.fft.ifft(f * np.conj(f)).real[:n]
    return np.sum(np.abs(ac))

def _greedy_local_search(x, max_iters=5000, deadline=None):
    """Greedy local search flipping single elements to reduce autoconv peak."""
    best = _autoconv_peak(x)
    x = x.copy()
    start = time.time()
    
    for _ in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        # Try flipping each element
        best_delta = 0
        best_idx = -1
        for i in range(N):
            x[i] = -x[i]
            val = _autoconv_peak(x)
            x[i] = -x[i]  # restore
            if val < best - 1e-9:
                best_delta = best - val
                best = val
                best_idx = i
                break  # accept first improvement (steepest descent)
        
        if best_idx == -1:
            break
        x[best_idx] = -x[best_idx]
    
    return x, best

def _simulated_annealing(x, deadline):
    """Simulated annealing to escape local minima."""
    x = x.copy()
    best_x = x.copy()
    best_val = _autoconv_peak(x)
    
    T = 1.0
    T_min = 0.001
    cooling = 0.995
    
    while time.time() < deadline - 2.0:
        if T < T_min:
            break
        
        # Random flip
        i = np.random.randint(N)
        old_val = best_val
        x[i] = -x[i]
        new_val = _autoconv_peak(x)
        delta = new_val - old_val
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            if new_val < best_val:
                best_val = new_val
                best_x = x.copy()
        else:
            x[i] = -x[i]  # reject
        
        T *= cooling
    
    return best_x, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s deadline, 20s safety margin
    
    # Start with Thue-Morse sequence
    x = _thue_morse(N).astype(np.float64)
    best_val = _autoconv_peak(x)
    best_x = x.copy()
    
    # Greedy local search
    x, val = _greedy_local_search(x, max_iters=2000, deadline=deadline - 60.0)
    if val < best_val:
        best_val = val
        best_x = x.copy()
    
    # Multiple restarts with simulated annealing
    restarts = 0
    while time.time() < deadline - 5.0:
        restarts += 1
        # Start from best found so far with some randomization
        x = best_x.copy()
        # Flip a few random elements
        n_flips = min(5, N // 2)
        for _ in range(n_flips):
            x[np.random.randint(N)] *= -1
        
        x, val = _simulated_annealing(x, deadline)
        if val < best_val:
            best_val = val
            best_x = x.copy()
        
        # If we've done enough restarts and time is running out, do a final greedy polish
        if time.time() > deadline - 10.0:
            break
    
    # Final greedy polish
    if time.time() < deadline - 1.0:
        x, val = _greedy_local_search(best_x, max_iters=5000, deadline=deadline - 0.5)
        if val < best_val:
            best_val = val
            best_x = x.copy()
    
    return best_x
# EVOLVE-BLOCK-END
