"""Find n in [10000, 1000000] minimizing the largest prime factor of n*(n+1)."""
import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    N_LOW = 10000
    N_HIGH = 1000000
    
    # We want to find n such that both n and n+1 have small largest prime factors.
    # The objective is min over n in [N_LOW, N_HIGH] of max(lpf(n), lpf(n+1)).
    
    # Strategy: Use a sieve to compute the largest prime factor (lpf) for all numbers
    # up to N_HIGH + 1. Then scan for the n that minimizes max(lpf[n], lpf[n+1]).
    
    # Sieve for largest prime factor
    max_val = N_HIGH + 1
    lpf = list(range(max_val + 1))
    
    # Sieve of Eratosthenes variant: for each prime p, mark multiples with p
    # lpf[x] will hold the largest prime factor of x
    for i in range(2, int(math.isqrt(max_val)) + 1):
        if lpf[i] == i:  # i is prime
            for j in range(i * i, max_val + 1, i):
                lpf[j] = i
    
    # For primes, lpf[p] = p (already set by initialization)
    # For composites, lpf[x] gets overwritten by larger primes as we go? 
    # Wait, no. The standard sieve for lpf:
    # Initialize lpf[i] = i for all i
    # For i from 2 to sqrt(max):
    #   if lpf[i] == i (i is prime):
    #     for j from i*i to max, step i:
    #       lpf[j] = i  -- this would give the SMALLEST prime factor
    # 
    # To get LARGEST prime factor, we need a different approach.
    # Option 1: Factor each number individually using trial division with precomputed primes.
    # Option 2: For each prime p, for each multiple of p, set lpf[multiple] = p. 
    #            If we iterate primes in increasing order, the last prime that divides x
    #            will be the largest. So we iterate p from 2 up, and for each multiple,
    #            we overwrite. The final value is the largest prime factor.
    
    # Let's redo the sieve properly for LARGEST prime factor:
    lpf = [0] * (max_val + 1)
    for i in range(2, max_val + 1):
        lpf[i] = i  # initialize
    
    # For each prime p (in increasing order), update multiples.
    # Since we go in increasing order, later (larger) primes will overwrite earlier ones.
    # But we only process i if lpf[i] == i (i is prime).
    for i in range(2, int(math.isqrt(max_val)) + 1):
        if lpf[i] == i:  # i is prime
            for j in range(i, max_val + 1, i):
                lpf[j] = i
    
    # Wait, this still gives the smallest prime factor because we overwrite with smaller i first?
    # No: we iterate i from 2 upward. When i=2, we set lpf[even] = 2.
    # When i=3, we set lpf[multiples of 3] = 3. This overwrites the 2 for numbers divisible by both 2 and 3.
    # So at the end, lpf[x] = the largest prime i (up to sqrt) that divides x? 
    # No, that's not right either. Let's think carefully.
    # 
    # Actually, the standard way: if we iterate primes in increasing order and do lpf[j] = p for each multiple,
    # then for a number x = p1 * p2 * ... * pk (p1 < p2 < ... < pk), 
    # when i=p1, lpf[x]=p1. When i=p2, lpf[x]=p2. ... When i=pk (if pk <= sqrt(max)), lpf[x]=pk.
    # But if pk > sqrt(max), then pk is never processed in the outer loop (since we only go to sqrt).
    # In that case, x has at most one prime factor > sqrt(x), and lpf[x] would retain the last small prime factor,
    # which is NOT the largest.
    #
    # Better approach: compute lpf by trial division using precomputed primes, or use a different sieve.
    #
    # Let's use a cleaner approach:
    # 1. Generate all primes up to max_val using standard sieve.
    # 2. For each number x, find its largest prime factor by dividing out primes.
    #
    # But that's O(N * pi(N)) which might be slow. 
    #
    # Alternative: Sieve where for each prime p, we mark lpf[multiple] = p for ALL multiples.
    # If we process primes in DECREASING order, the first (largest) prime that divides x wins.
    # Let's try: generate primes, then iterate in reverse.
    
    # Generate primes up to max_val
    is_prime = bytearray(b'\x01') * (max_val + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(math.isqrt(max_val)) + 1):
        if is_prime[i]:
            for j in range(i * i, max_val + 1, i):
                is_prime[j] = 0
    
    primes = [i for i in range(2, max_val + 1) if is_prime[i]]
    
    # Now compute lpf by iterating primes in decreasing order
    lpf = [0] * (max_val + 1)
    for p in reversed(primes):
        for j in range(p, max_val + 1, p):
            if lpf[j] == 0:
                lpf[j] = p
    
    # Check time
    if time.time() > deadline:
        # Fallback: return a reasonable default
        return 10000
    
    # Now scan for best n
    best_n = N_LOW
    best_val = float('inf')
    
    for n in range(N_LOW, N_HIGH + 1):
        val = max(lpf[n], lpf[n + 1])
        if val < best_val:
            best_val = val
            best_n = n
            # Early termination if we find a very good value
            if best_val <= 2:
                break
    
    if time.time() > deadline:
        return best_n
    
    return best_n
# EVOLVE-BLOCK-END
