"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import heapq

# EVOLVE-BLOCK-START

def _largest_prime_factor(n):
    """Return the largest prime factor of n. P(1)=1."""
    if n <= 1:
        return 1
    result = 1
    # factor out 2
    while n % 2 == 0:
        result = 2
        n //= 2
    # factor out odd numbers up to sqrt(n)
    f = 3
    while f * f <= n:
        while n % f == 0:
            result = f
            n //= f
        f += 2
    if n > 1:
        result = n
    return result

def construct():
    target = 123457
    deadline = time.time() + 80.0  # safety margin before 90s kill
    
    best_a = 1
    best_b = target - 1
    best_score = max(_largest_prime_factor(best_a), _largest_prime_factor(best_b))
    
    # We want to find a+b = target minimizing max(P(a), P(b)).
    # Strategy: iterate a from 1 to target//2, compute score, keep best.
    # But that's ~61728 iterations, each needing factorization.
    # Factorization of numbers up to 123457 is fast (sqrt ~ 351).
    # 61728 * ~350 divisions worst case = ~21M ops. Should be fine in 80s.
    
    # Actually, let's be smarter. 
    # P(a) is the largest prime factor. For a to have small P(a), a should be smooth.
    # We can generate smooth numbers and check.
    
    # Alternative: just iterate a from 1 to target//2, compute max(P(a), P(target-a)).
    # Since target is small (123457), brute force with efficient factorization should work.
    
    # Let's precompute smallest prime factor (SPF) for all numbers up to target.
    # Then P(n) can be computed quickly by walking the factorization.
    
    max_n = target
    spf = list(range(max_n + 1))  # smallest prime factor
    for i in range(2, int(math.isqrt(max_n)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, max_n + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def largest_pf(n):
        if n <= 1:
            return 1
        result = 1
        while n > 1:
            p = spf[n]
            result = p
            while n % p == 0:
                n //= p
        return result
    
    # Now iterate a from 1 to target//2
    half = target // 2
    for a in range(1, half + 1):
        b = target - a
        pa = largest_pf(a)
        pb = largest_pf(b)
        score = pa if pa > pb else pb
        if score < best_score:
            best_score = score
            best_a = a
            best_b = b
            if best_score == 1:
                break  # can't do better than 1
        if time.time() > deadline:
            break
    
    if best_a > best_b:
        best_a, best_b = best_b, best_a
    
    return [best_a, best_b]

# EVOLVE-BLOCK-END
