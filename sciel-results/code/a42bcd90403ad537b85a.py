import math
import random
import time

def is_prime_mr(n):
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

def sieve_primes(limit):
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(limit + 1) if sieve[i]]

def is_p_smooth(n, p, primes):
    """Check if n is p-smooth using trial division by primes up to p."""
    if n <= 1:
        return True
    remaining = n
    for q in primes:
        if q > p:
            break
        if q * q > remaining:
            break
        while remaining % q == 0:
            remaining //= q
    return remaining == 1

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin before 150s limit
    
    best_p = 7
    best_k = 1
    
    # Precompute primes up to a reasonable limit for trial division
    # We'll need primes up to max_p we search
    # Start with primes up to 10^6, extend if needed
    PRIME_LIMIT = 2000000
    primes = sieve_primes(PRIME_LIMIT)
    
    # Search primes in increasing order
    # For each prime p, check k=1,2,...,200
    # Use is_p_smooth with trial division
    
    # We can skip very small primes where k is likely small
    # But let's just iterate
    
    idx = 0
    while idx < len(primes):
        p = primes[idx]
        idx += 1
        if time.time() > deadline:
            break
        if p < 3:
            continue
        
        k = 0
        for j in range(1, 201):
            n = p * p + j
            if is_p_smooth(n, p, primes):
                k = j
            else:
                break
        
        if k > best_k:
            best_k = k
            best_p = p
    
    # If time remains, search larger primes
    if time.time() < deadline - 5:
        # Generate more primes in a segmented fashion
        last_prime = primes[-1] if primes else 2
        seg_size = 1000000
        seg_start = last_prime + 1
        while time.time() < deadline - 2:
            seg_end = seg_start + seg_size
            # Segmented sieve
            sieve = bytearray(b'\x01') * (seg_size + 1)
            for q in primes:
                if q * q > seg_end:
                    break
                start = ((seg_start + q - 1) // q) * q
                if start < seg_start:
                    start += q
                for i in range(start - seg_start, seg_size + 1, q):
                    sieve[i] = 0
            new_primes = [seg_start + i for i in range(seg_size + 1) if sieve[i]]
            # Filter with Miller-Rabin for certainty
            new_primes = [p for p in new_primes if is_prime_mr(p)]
            
            for p in new_primes:
                if time.time() > deadline:
                    break
                k = 0
                for j in range(1, 201):
                    n = p * p + j
                    if is_p_smooth(n, p, primes):
                        k = j
                    else:
                        break
                if k > best_k:
                    best_k = k
                    best_p = p
            
            # Extend our prime list for trial division
            if new_primes:
                primes.extend(new_primes)
                primes.sort()
            seg_start = seg_end + 1
    
    return {'p': best_p, 'k': best_k}
