import numpy as np
import time

N = 400

def _compute_psl(x):
    """
    Compute the peak sidelobe level (PSL) of the autocorrelation of x.
    x is a sequence of +/- 1.
    """
    n = len(x)
    size = 1
    while size < 2 * n:
        size *= 2
    X = np.fft.rfft(x, size)
    acf = np.fft.irfft(X * np.conj(X), size)[:n]
    # Sidelobes are lags 1 to n-1
    if n > 1:
        return np.max(np.abs(acf[1:]))
    return 0.0

def _simulated_annealing(x_init, time_limit, rng):
    """
    Perform Simulated Annealing to minimize PSL.
    """
    start_time = time.time()
    x = x_init.copy()
    current_psl = _compute_psl(x)
    best_psl = current_psl
    best_x = x.copy()
    
    # Parameters
    T_start = 5.0
    T_min = 0.001
    
    # We don't know exact steps/sec, so we just loop until time limit.
    # We need a cooling factor that works for a reasonable number of steps.
    # Let's assume ~5000 steps/sec. 10s -> 50,000 steps.
    # alpha = (T_min/T_start)^(1/50000) ~ 0.9998
    alpha = 0.9998
    
    while time.time() - start_time < time_limit:
        # Choose a move type
        move_type = rng.choice([1, 2, 3], p=[0.3, 0.3, 0.4])
        
        x_new = x.copy()
        
        if move_type == 1:
            idx = rng.randint(N)
            x_new[idx] *= -1
        elif move_type == 2:
            idx1, idx2 = rng.choice(N, size=2, replace=False)
            x_new[idx1] *= -1
            x_new[idx2] *= -1
        else:
            start_idx = rng.randint(N)
            length = rng.randint(5, 25)
            end_idx = min(start_idx + length, N)
            x_new[start_idx:end_idx] *= -1
            
        new_psl = _compute_psl(x_new)
        
        delta = new_psl - current_psl
        if delta < 0 or rng.random() < np.exp(-delta / T_start):
            x = x_new
            current_psl = new_psl
            if current_psl < best_psl:
                best_psl = current_psl
                best_x = x.copy()
        
        T_start *= alpha
        if T_start < T_min:
            # Reset temperature to escape local minima? 
            # Or just break and let construct() restart.
            # Breaking is cleaner for the restart strategy.
            break
            
    return best_x, best_psl

def construct():
    """
    Construct a signed step function (±1 sequence of length N) that minimizes
    the C3 autoconvolution upper bound (Peak Sidelobe Level).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
    rng = np.random.default_rng(42)
    
    best_x = None
    best_val = float('inf')
    
    # Phase 1: Multiple random restarts with Simulated Annealing
    while time.time() - start_time < deadline - 5.0:
        # Initialize with random sequence
        x_init = rng.choice([-1.0, 1.0], size=N)
        
        # Allocate time for this restart
        remaining = deadline - time.time()
        if remaining < 2.0:
            break
        
        # Give each SA run ~5-10 seconds
        t_alloc = min(8.0, remaining - 2.0)
        
        x_res, val = _simulated_annealing(x_init, t_alloc, rng)
        
        if val < best_val:
            best_val = val
            best_x = x_res.copy()
            
    # Fallback if no valid result was found
    if best_x is None:
        return np.ones(N)
        
    return best_x
