import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    
    MAX_N = 1000000
    MAX_VAL = MAX_N + 1  # we need LPF up to 1000001
    
    # Precompute largest prime factor for all numbers up to MAX_VAL
    # lpf[i] will store the largest prime factor of i
    lpf = [0] * (MAX_VAL + 1)
    
    # Sieve-like approach to find largest prime factor
    # Iterate p from 2 to MAX_VAL
    # If lpf[p] is 0, p is prime.
    # For all multiples m of p, lpf[m] = p.
    # Since p increases, the last prime factor assigned is the largest.
    for p in range(2, MAX_VAL + 1):
        if lpf[p] == 0:
            # p is prime
            for m in range(p, MAX_VAL + 1, p):
                lpf[m] = p
    
    # Check deadline
    if time.time() > deadline:
        return 10000

    # Step 3: Find n in [10000, 1000000] minimizing max(lpf[n], lpf[n+1])
    best_n = 10000
    best_val = max(lpf[10000], lpf[10001])
    
    for n in range(10000, MAX_N + 1):
        val = max(lpf[n], lpf[n+1])
        if val < best_val:
            best_val = val
            best_n = n
            if time.time() > deadline:
                break
    
    return best_n
# EVOLVE-BLOCK-END
