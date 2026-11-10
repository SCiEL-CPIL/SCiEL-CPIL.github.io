"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    MAX_N = 1000000
    MIN_N = 10000
    
    # Build smallest prime factor sieve up to MAX_N + 1
    # Using list of ints
    spf = list(range(MAX_N + 2))
    
    # Sieve of Eratosthenes
    limit = int(math.isqrt(MAX_N + 1))
    for i in range(2, limit + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, MAX_N + 2, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute powerful part for each number
    # powerful_part(n) = product of p^e where e >= 2 in factorization
    pp = [1] * (MAX_N + 2)
    
    # Optimization: Precompute powers? No, just loop.
    for n in range(MIN_N, MAX_N + 1):
        x = n
        val = 1
        while x > 1:
            p = spf[x]
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            if e >= 2:
                val *= (p ** e)
        pp[n] = val
        
        # Check time every 10000 iterations to avoid overhead
        if n % 10000 == 0 and time.time() > deadline:
            break
    
    # Find best n
    best_n = MIN_N
    best_val = 0
    # Ensure we only check up to MAX_N - 1
    end_range = MAX_N
    if time.time() > deadline - 5:
        # If we are close to deadline, just return best so far (which might be MIN_N if loop broke early)
        # But we should still try to find the best in the computed range
        pass
        
    for n in range(MIN_N, MAX_N):
        if n >= end_range:
            break
        # If pp[n+1] was not computed, it is 1.
        # If the loop broke early, pp[n+1] might be 1 for large n.
        # We should only consider n where pp[n+1] != 1 or n+1 < last_computed_n
        # But we don't track last_computed_n easily.
        # Assuming the loop finishes, this is fine.
        prod = pp[n] * pp[n + 1]
        if prod > best_val:
            best_val = prod
            best_n = n
            
    return best_n
# EVOLVE-BLOCK-END
