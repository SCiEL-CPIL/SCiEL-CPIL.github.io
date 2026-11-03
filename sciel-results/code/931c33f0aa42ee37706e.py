"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv(x):
    """Compute full autoconvolution of x using FFT."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size <<= 1
    X = np.fft.rfft(x, size)
    ac = np.fft.irfft(X * np.conj(X), size)[:2*n-1]
    return ac

def _objective(x):
    """Compute max |autocorr| / N^2 for a ±1 sequence."""
    ac = _autoconv(x)
    return np.max(np.abs(ac)) / (N * N)

def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Initialize with a good starting sequence
    # Use a Legendre-like sequence or random
    np.random.seed(42)
    x = np.random.choice([-1.0, 1.0], size=N)
    
    # Compute initial objective
    best_x = x.copy()
    best_obj = _objective(best_x)
    
    # Precompute autoconvolution for faster evaluation
    def _eval_flip(x, idx):
        """Evaluate objective if we flip x[idx], using incremental update."""
        # Flipping x[idx] changes autocorrelation at lags where x[idx] participates
        # ac[k] = sum_i x[i]*x[i+k]
        # Flipping x[idx] changes terms where i=idx or i+k=idx
        # This is complex; use FFT-based full recompute for correctness
        x_new = x.copy()
        x_new[idx] = -x_new[idx]
        ac = _autoconv(x_new)
        return np.max(np.abs(ac)) / (N * N)
    
    # Simulated annealing with in-place flips
    T_start = 1.0
    T_end = 0.001
    n_iterations = 15000
    
    # Track time
    t0 = time.time()
    
    current_obj = best_obj
    current_x = best_x.copy()
    
    # Precompute for speed: use a faster autocorrelation
    def _fast_autoconv(x):
        n = len(x)
        size = 1
        while size < 2 * n - 1:
            size <<= 1
        X = np.fft.rfft(x, size)
        power = X * np.conj(X)
        ac = np.fft.irfft(power, size)[:2*n-1]
        return ac
    
    current_x = best_x.copy()
    current_obj = best_obj
    
    # Simulated annealing
    for it in range(n_iterations):
        if time.time() > deadline:
            break
        
        # Pick random index to flip
        idx = np.random.randint(N)
        
        # Evaluate flip
        old_val = current_x[idx]
        current_x[idx] = -old_val
        new_obj = _objective(current_x)
        
        # Accept or reject
        delta = new_obj - current_obj
        if delta < 0 or np.random.random() < np.exp(-delta / max(T_start * (T_end/T_start)**(it/n_iterations), 1e-10)):
            current_obj = new_obj
            if new_obj < best_obj:
                best_obj = new_obj
                best_x = current_x.copy()
        else:
            current_x[idx] = old_val  # Revert
    
    # Greedy local search pass
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(N):
            if time.time() > deadline:
                break
            old_val = best_x[idx]
            best_x[idx] = -old_val
            new_obj = _objective(best_x)
            if new_obj < best_obj:
                best_obj = new_obj
                improved = True
            else:
                best_x[idx] = old_val
    
    return best_x
# EVOLVE-BLOCK-END
