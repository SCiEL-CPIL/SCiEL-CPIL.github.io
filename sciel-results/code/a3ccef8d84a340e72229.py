import math
import time
import numpy as np

# Deterministic Miller-Rabin for n < 2^64
def is_prime(n):
    if n < 2:
        return False
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False
    
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    
    for a in small_primes:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_k = 0
    best_p = 2
    
    # Sieve primes up to 2 * 10^6
    limit = 2 * 10**6
    is_prime_sieve = bytearray(b'\x01') * (limit + 1)
    is_prime_sieve[0] = 0
    is_prime_sieve[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime_sieve[i]:
            is_prime_sieve[i*i::i] = bytearray(len(is_prime_sieve[i*i::i]))
    
    primes = [i for i in range(2, limit + 1) if is_prime_sieve[i]]
    
    # For each candidate p, use batch smoothness check with numpy
    # p^2 + i for i=1..200 must all be p-smooth
    # Batch: for each prime q <= p, find which i in 1..200 have (p^2+i) % q == 0
    
    for p in primes:
        if time.time() > deadline:
            break
        
        p2 = p * p
        
        # Quick reject: if p^2+1 is prime, k=0
        if is_prime(p2 + 1):
            continue
        
        # Batch check: create array of values p^2+1 .. p^2+200
        # For each prime q <= p, divide out q from all values
        # After all divisions, if any value > 1, it's not p-smooth
        
        # Only do full check if p is large enough to be interesting
        # For small p, just do sequential check
        
        if p < 1000:
            # Sequential check for small p
            k = 0
            for i in range(1, 201):
                n_val = p2 + i
                temp = n_val
                for q in primes:
                    if q > p:
                        break
                    if q * q > temp:
                        break
                    while temp % q == 0:
                        temp //= q
                if temp == 1:
                    k = i
                else:
                    break
            if k > best_k:
                best_k = k
                best_p = p
            continue
        
        # For larger p, use batch approach
        # values[i] = p^2 + (i+1) for i=0..199
        values = np.array([p2 + i + 1 for i in range(200)], dtype=np.int64)
        
        # For each prime q <= p, divide out q from all values
        # We need primes up to p, but p can be up to 2*10^6
        # primes list has primes up to 2*10^6
        
        # Find index of last prime <= p
        # Binary search
        idx = 0
        lo, hi = 0, len(primes) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if primes[mid] <= p:
                idx = mid
                lo = mid + 1
            else:
                hi = mid - 1
        
        # Process primes up to index idx
        # For efficiency, only process primes where q <= p
        # For each q, find positions where values[i] % q == 0 and divide
        
        for j in range(idx + 1):
            q = primes[j]
            if q > p:
                break
            # Find indices where values[i] % q == 0
            # values[i] = p^2 + i + 1
            # (p^2 + i + 1) % q == 0 => i + 1 ≡ -p^2 (mod q)
            # i ≡ -p^2 - 1 (mod q)
            # i in [0, 199]
            rem = (-p2 - 1) % q
            # i = rem, rem+q, rem+2q, ... <= 199
            for i in range(rem, 200, q):
                # Divide values[i] by q as many times as possible
                while values[i] % q == 0:
                    values[i] //= q
        
        # After processing, check which values are 1
        # The longest prefix of 1s gives k
        k = 0
        for i in range(200):
            if values[i] == 1:
                k = i + 1
            else:
                break
        
        if k > best_k:
            best_k = k
            best_p = p
    
    return {'p': best_p, 'k': best_k}
