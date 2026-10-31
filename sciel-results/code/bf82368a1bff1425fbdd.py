"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _autoconv_bound(f):
    """Compute the autoconvolution upper bound (L1 norm of autoconvolution)."""
    # f is a 1D array of length N
    # Autoconvolution: c[k] = sum_i f[i] * f[i+k]
    # Using FFT for efficiency
    n = len(f)
    size = 2 * n
    F = np.fft.rfft(f, n=size)
    autoconv = np.fft.irfft(F * np.conj(F), n=size)[:n]
    # L1 norm of the autoconvolution
    return np.sum(np.abs(autoconv))

def _autoconv_max(f):
    """Compute the maximum absolute value of the autoconvolution."""
    n = len(f)
    size = 2 * n
    F = np.fft.rfft(f, n=size)
    autoconv = np.fft.irfft(F * np.conj(F), n=size)[:n]
    return np.max(np.abs(autoconv))

def _local_search(f, metric_fn, time_limit, seed=42):
    """Local search to minimize the metric."""
    rng = random.Random(seed)
    current = f.copy()
    current_score = metric_fn(current)
    best = current.copy()
    best_score = current_score
    
    start_time = time.time()
    deadline = start_time + time_limit
    
    n = len(current)
    
    while time.time() < deadline:
        # Try flipping a random position
        i = rng.randint(0, n - 1)
        new_f = current.copy()
        new_f[i] = -new_f[i]
        new_score = metric_fn(new_f)
        
        if new_score < current_score:
            current = new_f
            current_score = new_score
            if current_score < best_score:
                best = current.copy()
                best_score = current_score
        else:
            # Occasionally accept worse solutions to escape local minima
            if rng.random() < 0.01:
                current = new_f
                current_score = new_score
    
    return best, best_score

def construct():
    # Start with a random sign pattern
    rng = np.random.RandomState(42)
    f = rng.choice([-1.0, 1.0], size=N)
    
    # Try multiple seeds and metrics
    deadline = time.time() + 140  # Leave 10s safety margin
    
    best_f = f.copy()
    best_score = _autoconv_bound(f)
    
    # Try local search with L1 norm metric
    f1, score1 = _local_search(f, _autoconv_bound, 60, seed=1)
    if score1 < best_score:
        best_f = f1
        best_score = score1
    
    # Try with max metric
    f2, score2 = _local_search(f, _autoconv_max, 60, seed=2)
    if score2 < best_score:
        best_f = f2
        best_score = score2
    
    return best_f
# EVOLVE-BLOCK-END

