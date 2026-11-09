"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def construct():
    """
    Find integer n in [10000, 1000000] that minimizes the largest prime factor of n*(n+1).
    Uses an SPF sieve with early-exit pruning: once best_val is small,
    skip candidates where either n or n+1 has a prime factor >= best_val.
    """
    deadline = time.time() + 85  # 85s safety margin before 90s kill
    
    N = 1000000
    LOW = 10000
    limit = N + 1  # need factors up to 1000001
    
    # Build SPF sieve using NumPy for speed
    spf = np.arange(limit + 1, dtype=np.int64)
    spf[0] = 0
    spf[1] = 1
    
    for i in range(2, int(math.isqrt(limit)) + 1):
        if spf[i] == i:  # i is prime
            start = i * i
            # Vectorized assignment: for multiples of i starting from i*i
            # Only update where spf[j] == j (not yet assigned)
            mask = np.arange(start, limit + 1, i)
            # Filter to only those where spf is still identity
            for j in mask:
                if spf[j] == j:
                    spf[j] = i
    
    # Convert to list for faster scalar access in the tight loop
    spf_list = spf.tolist()
    
    # Function to get largest prime factor using SPF table, with early abort
    # if we find a factor >= threshold
    def largest_prime_factor_early(x, threshold):
        """Return LPF of x, or return threshold if any prime factor >= threshold is found."""
        if x <= 1:
            return 1
        lpf = 1
        while x > 1:
            p = spf_list[x]
            if p >= threshold:
                return threshold  # can't beat current best
            lpf = p
            while x % p == 0:
                x //= p
        return lpf
    
    # Full scan with early-exit pruning
    best_n = LOW
    best_val = float('inf')
    
    for n in range(LOW, N + 1):
        if time.time() > deadline:
            break
        
        # Early exit: if n has a prime factor >= best_val, skip
        lpf_n = largest_prime_factor_early(n, best_val)
        if lpf_n >= best_val:
            continue
        
        # Early exit: if n+1 has a prime factor >= best_val, skip
        lpf_np1 = largest_prime_factor_early(n + 1, best_val)
        if lpf_np1 >= best_val:
            continue
        
        val = lpf_n if lpf_n > lpf_np1 else lpf_np1
        if val < best_val:
            best_val = val
            best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
