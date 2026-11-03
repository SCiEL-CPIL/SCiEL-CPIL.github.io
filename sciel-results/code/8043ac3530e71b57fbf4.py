"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_score(x):
    """Compute max absolute autocorrelation / sum of squares."""
    n = len(x)
    # Compute full autocorrelation
    autocorr = np.correlate(x, x, mode='full')
    # autocorr is centered, length 2n-1
    # sum of squares is autocorr[n-1] (lag 0)
    sum_sq = np.sum(x**2)
    if sum_sq == 0:
        return float('inf')
    max_abs = np.max(np.abs(autocorr))
    return max_abs / sum_sq

def _generate_m_sequence(n_bits):
    """Generate a maximal-length sequence (m-sequence) using LFSR."""
    # Use a known primitive polynomial for m-sequences
    # For n=8, use x^8 + x^4 + x^3 + x^2 + 1 (0b11101 = 29)
    # For n=10, use x^10 + x^7 + 1 (0b100000101 = 261)
    # We'll use n=8 or n=10 and repeat/extend
    
    # Use a simple LFSR with a known primitive polynomial
    # n=8: taps at positions 0, 3, 4, 7 (0-indexed from LSB)
    # Polynomial: x^8 + x^4 + x^3 + x^2 + 1
    # Taps: 8, 4, 3, 2, 0 -> use taps [4, 3, 2] (excluding the top bit)
    
    def lfsr_sequence(length, n=8, taps=None):
        if taps is None:
            taps = [4, 3, 2]  # for x^8 + x^4 + x^3 + x^2 + 1
        # Initialize state
        state = 1  # non-zero initial state
        seq = []
        for _ in range(length):
            # Extract output bit
            out = state & 1
            seq.append(out)
            # Compute feedback
            fb = 0
            for t in taps:
                fb ^= (state >> t) & 1
            # Shift and add feedback
            state = (state >> 1) | (fb << (n - 1))
        return seq
    
    # Generate a long m-sequence and take what we need
    raw = lfsr_sequence(N, n=8, taps=[4, 3, 2])
    # Convert 0/1 to -1/+1
    x = np.array([1.0 if b == 1 else -1.0 for b in raw])
    return x

def _local_improve(x, max_iters=5000, deadline=None):
    """Local improvement by flipping individual elements."""
    n = len(x)
    current_score = _compute_score(x)
    best_x = x.copy()
    best_score = current_score
    x = x.copy()
    
    # Precompute autocorrelation efficiently
    def get_autocorr(x):
        return np.correlate(x, x, mode='full')
    
    autocorr = get_autocorr(x)
    
    for i in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try flipping a random position
        pos = np.random.randint(n)
        old_val = x[pos]
        x[pos] = -old_val
        
        new_score = _compute_score(x)
        if new_score < best_score:
            best_score = new_score
            best_x = x.copy()
            autocorr = get_autocorr(x)
        else:
            x[pos] = old_val  # revert
    
    return best_x, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Generate m-sequence based initial solution
    x = _generate_m_sequence(N)
    best_x = x.copy()
    best_score = _compute_score(x)
    
    # Local improvement
    best_x, best_score = _local_improve(x, max_iters=10000, deadline=deadline)
    
    # If time remains, try another random restart
    if time.time() < deadline - 5.0:
        # Try a random ±1 sequence with local improvement
        x2 = np.random.choice([-1.0, 1.0], size=N)
        x2, score2 = _local_improve(x2, max_iters=5000, deadline=deadline)
        if score2 < best_score:
            best_x = x2
            best_score = score2
    
    return best_x
# EVOLVE-BLOCK-END
