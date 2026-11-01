"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_max_sidelobe(s):
    """Compute the maximum absolute value of the autocorrelation sidelobes."""
    n = len(s)
    size = 2 * n
    f = np.fft.fft(s, size=size)
    power = np.abs(f) ** 2
    autocorr = np.fft.ifft(power).real
    sidelobes = np.concatenate([autocorr[1:], autocorr[-n+1:]])
    return np.max(np.abs(sidelobes))


def _local_search(s, rng, max_iter=10000):
    """Perform local search to improve the sequence."""
    current_s = s.copy()
    current_score = _compute_max_sidelobe(current_s)
    best_s = current_s.copy()
    best_score = current_score
    
    for _ in range(max_iter):
        idx = rng.integers(0, N)
        current_s[idx] = -current_s[idx]
        new_score = _compute_max_sidelobe(current_s)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_s = current_s.copy()
        else:
            current_s[idx] = -current_s[idx]  # revert
    
    return best_s, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    rng = np.random.default_rng(42)
    
    # Multiple random restarts
    best_s = None
    best_score = float('inf')
    
    # Try several random initializations
    n_restarts = 20
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        s = rng.choice([-1.0, 1.0], size=N)
        s, score = _local_search(s, rng, max_iter=5000)
        
        if score < best_score:
            best_score = score
            best_s = s
    
    if best_s is None:
        best_s = rng.choice([-1.0, 1.0], size=N)
    
    return best_s
# EVOLVE-BLOCK-END
