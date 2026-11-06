"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def sieve_primes(limit):
    """Generate all primes up to limit using Sieve of Eratosthenes."""
    is_prime = bytearray([1]) * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i:limit+1:i] = bytearray(len(is_prime[i*i:limit+1:i]))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def compute_phi_segment(start, length, primes):
    """
    Compute exact phi(n) for n in [start, start+length-1].
    """
    phi_val = [start + i for i in range(length)]
    rem = [start + i for i in range(length)]
    
    for p in primes:
        if start % p == 0:
            first = start
        else:
            first = start + (p - start % p)
            
        if first > start + length - 1:
            continue
            
        for multiple in range(first, start + length, p):
            idx = multiple - start
            phi_val[idx] = phi_val[idx] // p * (p - 1)
            while rem[idx] % p == 0:
                rem[idx] //= p
                
    for i in range(length):
        if rem[i] > 1:
            phi_val[i] = phi_val[i] // rem[i] * (rem[i] - 1)
            
    return phi_val

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Generate primes up to 10^6
    prime_limit = 10**6
    primes = sieve_primes(prime_limit)
    
    max_k = 200
    max_n = 10**12
    
    random.seed(42)
    
    # Try random segments
    for _ in range(1000):
        if time.time() > deadline:
            break
        s = random.randint(1, max_n - max_k + 1)
        phis = compute_phi_segment(s, max_k, primes)
        if len(set(phis)) == max_k:
            return {'start': s, 'length': max_k}
            
    # Fallback: try a fixed large number
    s = max_n - max_k + 1
    phis = compute_phi_segment(s, max_k, primes)
    if len(set(phis)) == max_k:
        return {'start': s, 'length': max_k}
        
    # Absolute fallback
    return {'start': 1, 'length': 1}

# EVOLVE-BLOCK-END
