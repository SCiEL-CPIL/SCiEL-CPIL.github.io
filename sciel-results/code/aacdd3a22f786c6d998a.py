import math
import time
import random
from collections import defaultdict

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
    
    for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
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

def pollard_rho(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    if n % 5 == 0:
        return 5
    
    x = 2
    y = 2
    d = 1
    c = 1
    while d == 1:
        x = (x * x + c) % n
        y = (y * y + c) % n
        y = (y * y + c) % n
        d = math.gcd(abs(x - y), n)
        if d == n:
            c += 1
            x = 2
            y = 2
            d = 1
            if c > 10:
                x = random.randint(2, n-2)
                y = x
                c = 1
    return d

# Precompute small primes for trial division
SMALL_PRIMES = []
def _gen_small_primes(limit):
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(limit + 1) if sieve[i]]

SMALL_PRIMES = _gen_small_primes(10000)

def is_p_smooth(n, p):
    """
    Returns True if all prime factors of n are <= p.
    """
    if n <= 1:
        return True
    if n <= p:
        return True
    
    # Trial division by small primes
    for q in SMALL_PRIMES:
        if q > p:
            break
        if q * q > n:
            break
        while n % q == 0:
            n //= q
        if n == 1:
            return True
        if n <= p:
            return True
            
    # If n is prime, check if it's <= p
    if is_prime(n):
        return n <= p
        
    # Use Pollard-Rho to find a factor
    d = pollard_rho(n)
    if d == n:
        return n <= p
        
    # Recursively check factors
    return is_p_smooth(d, p) and is_p_smooth(n // d, p)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_k = 0
    best_p = 2
    
    # Strategy: Sieve out candidate primes by checking quadratic residues
    # For each large prime q > p, if p^2 ≡ -i (mod q) for any i in [1,200],
    # then p^2 + i is divisible by q > p, so p^2+i is NOT p-smooth.
    # We want primes p where p^2 + i is NOT divisible by any q > p for i in [1,k].
    
    # Precompute: for each i in [1,200], find primes q where -i is a quadratic residue mod q
    # Then for such q, primes p where p^2 ≡ -i (mod q) are "bad" for that i.
    
    # Let's think differently: iterate through potential large prime factors q > p
    # For each q, find i in [1,200] such that -i is a quadratic residue mod q
    # Then p^2 ≡ -i (mod q) means p is a forbidden residue for that i
    
    # Actually, let's try a hybrid approach:
    # 1. Generate candidate primes up to some limit
    # 2. For each candidate p, quickly check if p^2+i has a factor > p using modular arithmetic
    # 3. Only do full factorization for promising candidates
    
    # Sieve primes up to 500,000 for search space
    limit = 500000
    is_prime_sieve = bytearray(b'\x01') * (limit + 1)
    is_prime_sieve[0] = is_prime_sieve[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime_sieve[i]:
            is_prime_sieve[i*i::i] = bytearray(len(is_prime_sieve[i*i::i]))
    
    primes = [i for i in range(2, limit + 1) if is_prime_sieve[i]]
    
    # For quick checking: precompute for each i in [1,200], which small primes q divide p^2+i
    # But that's expensive. Instead, let's just try more primes with better filtering.
    
    # Better approach: for each p, check p^2+i for i=1,2,3,... and stop when we find
    # a factor > p. Use modular arithmetic to quickly detect large factors.
    
    for p in primes:
        if time.time() > deadline:
            break
            
        k = 0
        p_sq = p * p
        
        for i in range(1, 201):
            if time.time() > deadline:
                break
                
            n_val = p_sq + i
            
            # Quick check: does n_val have a prime factor > p?
            # If n_val is prime and > p, it's not p-smooth
            # If n_val has a factor > p, it's not p-smooth
            
            # First, check if n_val is prime
            if is_prime(n_val):
                if n_val > p:
                    break
                else:
                    k = i
                    continue
            
            # Check for small factors first
            has_large_factor = False
            temp_n = n_val
            
            # Trial division by primes up to min(p, sqrt(n_val))
            for q in SMALL_PRIMES:
                if q > p:
                    break
                if q * q > temp_n:
                    break
                while temp_n % q == 0:
                    temp_n //= q
                if temp_n == 1:
                    has_large_factor = False
                    break
                if temp_n <= p:
                    # Remaining factor is <= p, so it's p-smooth so far
                    has_large_factor = False
                    break
            
            if has_large_factor:
                break
            
            # If temp_n > 1 and temp_n > p, we need to check if it's prime or has large factors
            if temp_n > 1:
                if temp_n > p:
                    # Check if temp_n is prime
                    if is_prime(temp_n):
                        break
                    else:
                        # Factor temp_n and check for factors > p
                        d = pollard_rho(temp_n)
                        if d == temp_n:
                            # Should be prime, but just in case
                            if temp_n > p:
                                break
                        else:
                            # Check both factors
                            if d > p or (temp_n // d) > p:
                                break
                            # Recursively check
                            if not is_p_smooth(d, p) or not is_p_smooth(temp_n // d, p):
                                break
            
            k = i
            
        if k > best_k:
            best_k = k
            best_p = p
            
        if best_k == 200:
            break

    return {'p': best_p, 'k': best_k}
