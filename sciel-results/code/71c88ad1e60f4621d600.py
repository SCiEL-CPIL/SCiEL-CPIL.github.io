"""Smooth two-summand decomposition: minimize max(P(a), P(b)) for a+b=123457."""
import time
import math

TARGET = 123457

def construct():
    """
    Return [a, b] with a + b = 123457 minimizing max(P(a), P(b)),
    where P(n) is the largest prime factor of n (P(1)=1).
    
    Strategy: 
    1. Generate primes up to TARGET.
    2. Incrementally build the set of B-smooth numbers (numbers whose prime factors
       are all <= B) as B increases through the primes.
    3. At each step, check if any pair of B-smooth numbers sums to TARGET.
       The first such pair found has the minimal possible max(P(a), P(b)) = B.
    4. If no such pair is found within the time budget, fall back to a full scan
       using a precomputed largest-prime-factor array.
    """
    deadline = time.time() + 80.0  # safety margin before 90s kill
    
    # Step 1: Sieve to generate primes up to TARGET
    is_prime = [True] * (TARGET + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(TARGET)) + 1):
        if is_prime[i]:
            for j in range(i * i, TARGET + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, TARGET + 1) if is_prime[i]]
    
    # Step 2: Incrementally build smooth number set and check for a valid pair
    # smooth_set contains all numbers whose prime factors are from primes[0:idx+1]
    smooth_set = set([1])
    best_pair = (1, TARGET - 1)
    found_optimal = False
    
    for idx, p in enumerate(primes):
        if time.time() > deadline:
            break
        
        # Add all new smooth numbers that include prime p
        # For each existing smooth number r, add r*p, r*p^2, ... <= TARGET
        current_list = list(smooth_set)
        for r in current_list:
            val = r * p
            while val <= TARGET:
                smooth_set.add(val)
                val *= p
        
        # Check if any a in smooth_set has TARGET - a also in smooth_set
        # We want a >= 1, b = TARGET - a >= 1, so 1 <= a <= TARGET - 1
        for a in smooth_set:
            if a < 1 or a > TARGET - 1:
                continue
            b = TARGET - a
            if b in smooth_set:
                best_pair = (a, b)
                found_optimal = True
                break
        
        if found_optimal:
            # We found the minimal B such that both a and b are B-smooth
            # This is the global optimum: max(P(a), P(b)) = p
            break
    
    if found_optimal:
        return [best_pair[0], best_pair[1]]
    
    # Step 3: Fallback — full scan with precomputed largest prime factor
    # Build lp[n] = largest prime factor of n for all n up to TARGET
    lp = [0] * (TARGET + 1)
    for p in primes:
        for m in range(p, TARGET + 1, p):
            lp[m] = p
    lp[1] = 1
    
    best_max = float('inf')
    for a in range(1, TARGET):
        if time.time() > deadline:
            break
        b = TARGET - a
        pa = lp[a]
        pb = lp[b]
        mx = pa if pa > pb else pb
        if mx < best_max:
            best_max = mx
            best_pair = (a, b)
            if best_max <= 2:
                break
    
    return [best_pair[0], best_pair[1]]
