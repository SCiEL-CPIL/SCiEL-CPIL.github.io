"""
Optimized smooth two-summand decomposition.
Uses a sieve to compute the largest prime factor for all integers up to 123457,
then performs a linear scan to directly identify the global minimum of max(P(a), P(b)).
"""
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 90s kill
    TARGET = 123457
    
    # Step 1: Sieve of Eratosthenes to get all primes up to TARGET
    is_prime = [True] * (TARGET + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(TARGET**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, TARGET + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, TARGET + 1) if is_prime[i]]
    
    # Step 2: Compute the largest prime factor P(n) for each n in [1, TARGET]
    # P(1) = 1
    largest_pf = [1] * (TARGET + 1)
    
    # Process primes in increasing order: for each prime p, mark all multiples.
    # Since we go in increasing order, the last prime that divides n gives P(n).
    for p in primes:
        for m in range(p, TARGET + 1, p):
            largest_pf[m] = p
    
    # Check time
    if time.time() > deadline:
        return [1, TARGET - 1]
    
    # Step 3: Linear scan over all pairs (a, b) where a + b = TARGET
    # We want to minimize max(P(a), P(b))
    best_max = TARGET + 1  # Upper bound
    best_a = 1
    best_b = TARGET - 1
    
    for a in range(1, TARGET):
        b = TARGET - a
        pa = largest_pf[a]
        pb = largest_pf[b]
        current_max = pa if pa > pb else pb
        if current_max < best_max:
            best_max = current_max
            best_a = a
            best_b = b
            # Early termination: minimum possible max is 2 (if one is 1 and other is power of 2)
            # But we can't guarantee that, so continue to find true minimum
            if best_max == 2:
                break
    
    return [best_a, best_b]
