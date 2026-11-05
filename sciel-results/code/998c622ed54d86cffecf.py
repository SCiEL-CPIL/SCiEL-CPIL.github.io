"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 115.0  # Safety margin before 120s kill

def _compute_c3_upper_bound(f):
    """
    Compute the upper bound on C3 for a signed step function f with N intervals.
    Returns max |autoconvolution| excluding the zero-lag self term.
    """
    n = len(f)
    size = 2 * n
    F = np.fft.rfft(f, n=size)
    conv = np.fft.irfft(F * F, n=size)
    conv_lags = conv[:n]
    if len(conv_lags) > 1:
        return np.max(np.abs(conv_lags[1:]))
    return 0.0

def _thue_morse(n):
    """Generate the first n terms of the Thue-Morse sequence (0/1)."""
    t = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        t[i] = t[i >> 1] ^ (i & 1)
    return t

def _to_signed(t, polarity=1):
    """Convert 0/1 Thue-Morse to signed ±1. polarity=1: 0->+1,1->-1; polarity=-1: 0->-1,1->+1"""
    if polarity == 1:
        return 1.0 - 2.0 * t.astype(float)
    else:
        return -1.0 + 2.0 * t.astype(float)

def _local_search(f, time_limit):
    """Greedy local search: flip individual bits to reduce the objective."""
    n = len(f)
    best_f = f.copy()
    best_score = _compute_c3_upper_bound(best_f)
    deadline = time.time() + time_limit
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(n):
            old_val = best_f[idx]
            best_f[idx] = -old_val
            new_score = _compute_c3_upper_bound(best_f)
            if new_score < best_score - 1e-12:
                best_score = new_score
                improved = True
            else:
                best_f[idx] = old_val
    return best_f, best_score

def construct():
    """
    Main entry point. Returns a sequence of N signed step heights.
    Score is the upper bound on C3 (lower is better).
    """
    start_time = time.time()
    deadline = start_time + _DEADLINE

    # Generate Thue-Morse sequence of length N
    t = _thue_morse(N)
    
    # Try both polarities
    candidates = []
    for polarity in [1, -1]:
        f = _to_signed(t, polarity)
        score = _compute_c3_upper_bound(f)
        candidates.append((score, f.copy()))
    
    # Sort by score
    candidates.sort(key=lambda x: x[0])
    best_score, best_f = candidates[0]
    
    # Local search on the best candidate
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        best_f, best_score = _local_search(best_f, remaining_time - 5)
    
    # If time remains, try local search on second best too
    if len(candidates) > 1 and time.time() < deadline - 5:
        remaining = deadline - time.time() - 2
        if remaining > 1:
            f2, score2 = candidates[1]
            f2_opt, score2_opt = _local_search(f2, remaining)
            if score2_opt < best_score:
                best_f = f2_opt
                best_score = score2_opt
    
    return best_f

# EVOLVE-BLOCK-END
