"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from math import isqrt

def _sieve_primes(n):
    """Return list of primes up to n using simple sieve."""
    if n < 2:
        return []
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]

def _sigma_prime_power(p, e):
    """Compute sigma(p^e) = (p^(e+1)-1)/(p-1)."""
    if e == 0:
        return 1
    # Compute p^(e+1) - 1
    val = 1
    for _ in range(e + 1):
        val *= p
    return (val - 1) // (p - 1)

def _factorize(n, primes):
    """Factorize n using given primes. Returns list of (p, e)."""
    factors = []
    temp = n
    for p in primes:
        if p * p > temp:
            break
        if temp % p == 0:
            e = 0
            while temp % p == 0:
                temp //= p
                e += 1
            factors.append((p, e))
    if temp > 1:
        factors.append((temp, 1))
    return factors

def _sigma_from_factors(factors):
    """Compute sigma from factorization list."""
    result = 1
    for p, e in factors:
        result *= _sigma_prime_power(p, e)
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    MAX_N = 10_000_000
    target_ratio = 3.0 / 2.0
    
    # Generate primes up to sqrt(MAX_N) for factorization
    primes = _sieve_primes(isqrt(MAX_N) + 1)
    
    # Strategy: Search for pairs (a, b) with a/b close to 3/2
    # and sigma(a) = sigma(b)
    
    # Approach: For each candidate ratio near 3/2, generate pairs and check sigma equality
    # We'll search b in a range and compute a = round(3*b/2), then check nearby values
    
    best_error = float('inf')
    best_pair = (11, 6)  # Default
    
    # First, do a quick check of the known pair
    # sigma(11) = 12, sigma(6) = 12. 11/6 = 1.833..., |11/6 - 3/2| = |1.833... - 1.5| = 0.333...
    
    # Let's search more systematically
    # For b from 1 to MAX_N, a should be near 3*b/2
    # But we need both a and b <= MAX_N, so b <= MAX_N and a = 3b/2 <= MAX_N => b <= 2*MAX_N/3
    
    max_b = min(MAX_N, 2 * MAX_N // 3)
    
    # We'll search in chunks and check time
    # For each b, consider a in a small window around 3*b/2
    
    # To make this efficient, let's precompute sigma for all numbers up to MAX_N
    # Using a sieve method
    
    sigma_arr = [0] * (MAX_N + 1)
    for i in range(1, MAX_N + 1):
        for j in range(i, MAX_N + 1, i):
            sigma_arr[j] += i
    
    # Now sigma_arr[n] = sigma(n) for all n
    # Search for pairs with sigma(a) == sigma(b) and a/b close to 3/2
    
    # For each b, a_target = 3*b/2
    # Check a in [a_target - delta, a_target + delta] for small delta
    
    # Let's be smart: iterate b, compute target_a = 3*b//2, check neighbors
    # and also check if sigma matches
    
    # We'll search with increasing window size
    for window in range(1, 100):
        if time.time() > deadline - 5.0:
            break
        
        for b in range(1, max_b + 1, 1):
            if time.time() > deadline - 5.0:
                break
            
            target_a_float = 3.0 * b / 2.0
            a_low = max(1, int(target_a_float - window))
            a_high = min(MAX_N, int(target_a_float + window))
            
            for a in range(a_low, a_high + 1):
                if a == b:
                    continue
                if sigma_arr[a] == sigma_arr[b]:
                    # Compute error
                    ratio = a / b
                    error = abs(ratio - target_ratio)
                    if error < best_error:
                        best_error = error
                        best_pair = (a, b)
    
    # Also try the reverse: for each a, find b near 2*a/3
    if time.time() < deadline - 5.0:
        for window in range(1, 100):
            if time.time() > deadline - 5.0:
                break
            
            for a in range(1, MAX_N + 1, 1):
                if time.time() > deadline - 5.0:
                    break
                
                target_b_float = 2.0 * a / 3.0
                b_low = max(1, int(target_b_float - window))
                b_high = min(MAX_N, int(target_b_float + window))
                
                for b in range(b_low, b_high + 1):
                    if a == b:
                        continue
                    if sigma_arr[a] == sigma_arr[b]:
                        ratio = a / b
                        error = abs(ratio - target_ratio)
                        if error < best_error:
                            best_error = error
                            best_pair = (a, b)
    
    a, b = best_pair
    # Ensure they are Python ints
    return [int(a), int(b)]

# EVOLVE-BLOCK-END
