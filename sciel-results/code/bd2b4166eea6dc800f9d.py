import numpy as np
import time

N = 400

def _compute_sidelobe_sum(x):
    """
    Compute the sum of absolute values of the autocorrelation sidelobes.
    x is a 1D array of length N.
    """
    # Compute autocorrelation using FFT
    # Pad to next power of 2 for efficiency, though 400 is small
    n_fft = 1 << (2 * N - 1).bit_length()
    X = np.fft.rfft(x, n=n_fft)
    power = X * np.conjugate(X)
    r = np.fft.irfft(power, n=n_fft)
    # The autocorrelation R(k) for k=0..N-1 is in the first N samples of r
    # r[0] is R(0) = sum(x^2)
    # r[1] is R(1)
    # ...
    # r[N-1] is R(N-1)
    # Due to symmetry, R(-k) = R(k).
    # The sum of absolute sidelobes is sum_{k=1}^{N-1} |R(k)|.
    # Note: irfft returns real values.
    sidelobes = r[1:N]
    return np.sum(np.abs(sidelobes))

def _local_search(x, max_passes=100, deadline=None):
    """
    Perform local search by flipping bits that reduce the objective.
    """
    x = x.copy()
    current_obj = _compute_sidelobe_sum(x)
    
    for p in range(max_passes):
        if deadline and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline and time.time() > deadline:
                break
            # Flip bit i
            old_val = x[i]
            x[i] = -old_val
            new_obj = _compute_sidelobe_sum(x)
            if new_obj < current_obj:
                current_obj = new_obj
                improved = True
            else:
                x[i] = old_val
        if not improved:
            break
    return x, current_obj

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    best_x = None
    best_obj = np.inf
    
    # Use multiple random restarts to find a better local minimum
    # N=400 is small, so we can afford many restarts
    rng = np.random.default_rng(42)
    
    num_restarts = 50
    max_passes_per_restart = 20
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize with random +/- 1 sequence
        x = rng.choice([-1, 1], size=N)
        
        # Run local search
        x_opt, obj_opt = _local_search(x, max_passes=max_passes_per_restart, deadline=deadline)
        
        if obj_opt < best_obj:
            best_obj = obj_opt
            best_x = x_opt.copy()
            
        # If we find a very good solution, we might spend more time polishing it
        # But for now, simple restarts work well.
        
    if best_x is None:
        # Fallback to random if no time left
        best_x = rng.choice([-1, 1], size=N)
        
    return best_x
