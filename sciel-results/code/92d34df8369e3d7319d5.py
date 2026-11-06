"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Find n in [10000, 1000000] that minimizes the largest prime factor of n(n+1).
    
    Strategy: Precompute the largest prime factor (LPF) for every integer up to
    1,000,001 using a vectorized sieve. Then scan all candidates n in [10000, 1000000]
    using vectorized operations to find the minimum of max(lpf[n], lpf[n+1]).
    """
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline with 5s safety margin
    
    N_MIN = 10000
    N_MAX = 1000000
    N_MAX_PLUS_1 = N_MAX + 1
    SIZE = N_MAX_PLUS_1 + 1
    
    # Precompute largest prime factor for all numbers up to N_MAX_PLUS_1
    lpf = np.zeros(SIZE, dtype=np.int32)
    
    # Sieve: for each prime p, set lpf[p] = p, and for multiples, update lpf
    # Using vectorized assignment for speed
    for p in range(2, SIZE):
        if lpf[p] == 0:  # p is prime
            lpf[p:SIZE:p] = p
    
    # Scan all candidates using vectorized max and argmin
    # lpf[N_MIN : N_MAX+1] corresponds to n
    # lpf[N_MIN+1 : N_MAX+2] corresponds to n+1
    # Note: SIZE is N_MAX+2, so index N_MAX+1 is the last valid index.
    # Slice end is exclusive, so N_MAX+2 is correct.
    
    # Ensure we don't go out of bounds if N_MAX+1 is the last index
    # lpf has indices 0..SIZE-1. SIZE-1 = N_MAX+1.
    # We need lpf[N_MAX+1].
    
    vals = np.maximum(lpf[N_MIN:N_MAX+1], lpf[N_MIN+1:N_MAX+2])
    
    # Find the index of the minimum value
    min_idx = np.argmin(vals)
    best_n = N_MIN + min_idx
    
    return int(best_n)
# EVOLVE-BLOCK-END
