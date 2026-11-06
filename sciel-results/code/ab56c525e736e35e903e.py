"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N_MIN = 10000
    N_MAX = 1000000
    LIMIT = N_MAX + 1  # We need lpf up to N_MAX
    
    # Sieve to compute largest prime factor for all numbers up to LIMIT
    lpf = np.zeros(LIMIT + 1, dtype=np.int64)
    
    # Initialize: for each i, lpf[i] will be the largest prime factor
    # Use a sieve approach: for each prime p, mark multiples
    
    # First, find primes using a simple sieve
    is_prime = np.ones(LIMIT + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    
    for i in range(2, int(np.sqrt(LIMIT)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    primes = np.where(is_prime)[0]
    
    # For each prime, update lpf for its multiples
    # To get the LARGEST prime factor, we iterate primes in increasing order
    # and for each multiple, set lpf[multiple] = p
    # Since we go in increasing order, the last prime that divides a number is the largest
    
    # But this is inefficient if done naively. Let's think differently.
    # For each number n, lpf[n] = largest prime p such that p divides n.
    
    # Alternative: initialize lpf to 0. For each prime p from smallest to largest,
    # for each multiple m of p, set lpf[m] = p. At the end, lpf[m] is the largest prime factor.
    
    # This is O(n log log n) roughly.
    
    for p in primes:
        for multiple in range(p, LIMIT + 1, p):
            lpf[multiple] = p
    
    # Now lpf[n] is the largest prime factor of n for n in [2, LIMIT]
    # lpf[0] and lpf[1] are 0, which is fine since n >= 10000
    
    # Now find n in [N_MIN, N_MAX] that minimizes max(lpf[n], lpf[n+1])
    
    # Vectorized computation
    # For each n in [N_MIN, N_MAX], compute max(lpf[n], lpf[n+1])
    
    n_vals = np.arange(N_MIN, N_MAX + 1)
    lpf_n = lpf[n_vals]
    lpf_n1 = lpf[n_vals + 1]
    
    max_lpf = np.maximum(lpf_n, lpf_n1)
    
    # Find the minimum value
    min_max_lpf = np.min(max_lpf)
    
    # Find all n that achieve this minimum
    best_indices = np.where(max_lpf == min_max_lpf)[0]
    
    if len(best_indices) > 0:
        # Return the first one (or any)
        best_n = int(n_vals[best_indices[0]])
        return best_n
    else:
        # Fallback
        return N_MIN

# EVOLVE-BLOCK-END
