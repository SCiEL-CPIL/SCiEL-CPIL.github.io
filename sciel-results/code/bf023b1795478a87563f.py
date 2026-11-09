"""Signed step functions minimizing max|autoconv| / (sum)^2"""
import numpy as np
import time

N = 400
DEADLINE = 140  # seconds, safety margin before 150s kill

# EVOLVE-BLOCK-START

def _autoconv_full(f):
    """Compute full autoconvolution using FFT for speed."""
    n = len(f)
    size = 2 * n
    F = np.fft.rfft(f, size=size)
    ac = np.fft.irfft(F * np.conj(F), size=size)[:n]
    return ac

def _objective(f):
    """Compute max|autoconv| / (sum(f))^2. Return inf if sum is 0."""
    S = f.sum()
    if abs(S) < 1e-10:
        return np.inf
    ac = _autoconv_full(f)
    max_abs = np.max(np.abs(ac))
    return max_abs / (S * S)

def _try_flip(f, i, current_obj):
    """Try flipping sign at index i, return new objective."""
    f[i] = -f[i]
    obj = _objective(f)
    f[i] = -f[i]  # revert
    return obj

def _local_search(f, deadline):
    """Greedy local search: flip signs to reduce objective."""
    f = f.copy()
    current_obj = _objective(f)
    if not np.isfinite(current_obj):
        return f, current_obj
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_obj = current_obj
        best_pos = -1
        for i in range(N):
            if time.time() > deadline:
                break
            new_obj = _try_flip(f, i, current_obj)
            if new_obj < best_obj - 1e-12:
                best_obj = new_obj
                best_pos = i
        if best_pos >= 0:
            f[best_pos] = -f[best_pos]
            current_obj = best_obj
            improved = True
    
    return f, current_obj

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_f = None
    best_obj = np.inf
    
    # Start with all-ones
    f0 = np.ones(N, dtype=np.float64)
    f0, obj0 = _local_search(f0, deadline)
    best_f = f0.copy()
    best_obj = obj0
    
    # Multi-start with random initializations biased toward +1
    rng = np.random.default_rng(42)
    n_starts = 20
    for s in range(n_starts):
        if time.time() > deadline:
            break
        p_flip = 0.1 + 0.05 * s
        f_init = np.ones(N, dtype=np.float64)
        flip_mask = rng.random(N) < p_flip
        f_init[flip_mask] = -1.0
        
        f_trial, obj_trial = _local_search(f_init, deadline)
        if obj_trial < best_obj:
            best_obj = obj_trial
            best_f = f_trial.copy()
    
    # Final polish: try a few more random restarts if time permits
    if time.time() < deadline - 10:
        for s in range(10):
            if time.time() > deadline:
                break
            f_init = rng.choice([-1.0, 1.0], size=N)
            f_trial, obj_trial = _local_search(f_init, deadline)
            if obj_trial < best_obj:
                best_obj = obj_trial
                best_f = f_trial.copy()
    
    return best_f

# EVOLVE-BLOCK-END
