import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Find n in [10000, 1000000] that minimizes the largest prime factor of n(n+1).
    This is equivalent to minimizing max(LPF(n), LPF(n+1)).
    
    Strategy:
    1. Precompute the Largest Prime Factor (LPF) for all numbers up to 1,000,001
       using a sieve method.
    2. Iterate through the valid range of n and find the minimum of the max LPF.
    """
    start_time = time.time()
    # Safety deadline: 50 seconds (limit is 60s, but we should be much faster)
    deadline = start_time + 50.0

    N_MIN = 10000
    N_MAX = 1000000
    LIMIT = N_MAX + 1 # We need LPF up to 1000001

    # Initialize LPF array
    # Using int32 is sufficient for values up to 10^6
    lpf = np.ones(LIMIT + 1, dtype=np.int32)
    
    # Sieve for Largest Prime Factor
    # Iterate through potential primes p
    for p in range(2, LIMIT + 1):
        # If lpf[p] is still 1, p is a prime (hasn't been marked by a smaller prime)
        if lpf[p] == 1:
            # Mark all multiples of p with p
            # Since we iterate p in increasing order, the last assignment
            # for any number j will be from its largest prime factor.
            lpf[p::p] = p
            
        # Check time periodically (every 10000 iterations to avoid overhead)
        if p % 10000 == 0 and time.time() > deadline:
            # Fallback to a reasonable default if timeout (unlikely)
            return 10000

    # We need to minimize max(lpf[n], lpf[n+1]) for n in [N_MIN, N_MAX]
    # lpf[n] is at index n
    # lpf[n+1] is at index n+1
    
    # Extract slices
    # n ranges from N_MIN to N_MAX
    # lpf[n] ranges from lpf[N_MIN] to lpf[N_MAX]
    # lpf[n+1] ranges from lpf[N_MIN+1] to lpf[N_MAX+1]
    
    a = lpf[N_MIN:N_MAX+1]
    b = lpf[N_MIN+1:N_MAX+2]
    
    # Compute max for each pair
    m = np.maximum(a, b)
    
    # Find the index of the minimum value
    min_idx = np.argmin(m)
    
    # Map index back to n
    best_n = N_MIN + min_idx
    
    return int(best_n)
# EVOLVE-BLOCK-END
