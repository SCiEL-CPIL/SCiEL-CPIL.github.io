import time

# EVOLVE-BLOCK-START
def construct():
    # Safety margin: The evaluator kills after 90s.
    # We use 85s as our hard stop to be safe.
    deadline = time.time() + 85
    
    MAX_N = 1000000
    MAX_VAL = MAX_N + 1  # We need lpf up to 1000001
    
    # Precompute largest prime factor (lpf) for all numbers up to MAX_VAL
    # lpf[i] will store the largest prime factor of i
    # Initialize with 0. 0 indicates "not yet processed" or "prime itself" logic later.
    lpf = [0] * (MAX_VAL + 1)
    
    # Sieve of Eratosthenes variant for Largest Prime Factor
    # Iterate p from 2 to MAX_VAL
    # If lpf[p] is 0, p is prime (no smaller prime factor found)
    # For all multiples m of p, set lpf[m] = p
    # Since p increases, the last assignment is the largest prime factor.
    for p in range(2, MAX_VAL + 1):
        if lpf[p] == 0:
            # p is prime
            # Mark all multiples
            for m in range(p, MAX_VAL + 1, p):
                lpf[m] = p
                
    # Check if we are running out of time before the scan
    if time.time() > deadline:
        # Fallback to a known good value if sieve took too long (unlikely)
        return 10000

    # Find n in [10000, 1000000] minimizing max(lpf[n], lpf[n+1])
    best_n = 10000
    best_val = max(lpf[10000], lpf[10001])
    
    # Local variable lookup optimization
    lpf_local = lpf
    deadline_local = deadline
    time_local = time.time
    
    for n in range(10000, MAX_N + 1):
        val = lpf_local[n]
        if lpf_local[n+1] > val:
            val = lpf_local[n+1]
            
        if val < best_val:
            best_val = val
            best_n = n
            
            # Periodic time check
            if time_local() > deadline_local:
                break

    return best_n
# EVOLVE-BLOCK-END
