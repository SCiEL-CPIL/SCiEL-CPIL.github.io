import math
import random
import time

def is_prime(n):
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
    for a in [2, 3, 5, 7, 11]:
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

def pollard_rho(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    if n % 5 == 0:
        return 5
    while True:
        x = random.randint(2, n - 1)
        y = x
        c = random.randint(1, n - 1)
        d = 1
        while d == 1:
            x = (x * x + c) % n
            y = (y * y + c) % n
            y = (y * y + c) % n
            d = math.gcd(abs(x - y), n)
        if d != n:
            return d

def is_p_smooth(n, p, small_primes):
    """
    Checks if all prime factors of n are <= p.
    Uses trial division by small_primes first.
    Then checks the remaining cofactor.
    """
    if n == 1:
        return True
    
    # 1. Trial division by small primes
    m = n
    for q in small_primes:
        if q * q > m:
            break
        if q > p:
            # If q > p and m % q == 0, then m has a factor > p.
            # But wait, if q > p, we should have stopped earlier?
            # No, small_primes are up to B. If B > p, we need to be careful.
            # If q > p, and m % q == 0, then q is a factor > p, so return False.
            # However, if q > p, it's a prime factor > p.
            if m % q == 0:
                return False
            continue
        
        if m % q == 0:
            while m % q == 0:
                m //= q
            if m == 1:
                return True
    
    if m == 1:
        return True
    
    # 2. Check the remaining cofactor m
    # If m < p^2, then m is either prime or a product of two primes.
    # Actually, if m is prime, it must be <= p.
    # If m is composite, it has a factor <= sqrt(m).
    
    if m <= p:
        # m is a prime (or 1) and <= p. 
        # Wait, if m is composite, it has factors < m <= p. So it's smooth.
        # Is m necessarily prime? No. But if m <= p, all its factors are < m <= p.
        # So m is p-smooth.
        return True
    
    # If m > p:
    # If m is prime, it is > p, so not smooth.
    if is_prime(m):
        return False
    
    # If m is composite, find a factor.
    # If any factor > p, not smooth.
    # Since m > p, and m is composite, it has a factor d <= sqrt(m).
    # If d <= p, then m/d might be > p.
    # We need to ensure ALL factors are <= p.
    
    d = pollard_rho(m)
    # Recursively check d and m//d
    # But we can just check if d <= p and m//d is p-smooth.
    # Since m is not too large (p^2 + k), m//d is also not huge.
    
    if d > p:
        return False
    # Check m//d
    return is_p_smooth(m // d, p, small_primes)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    best_p = 7
    best_k = 1
    
    # Generate small primes for trial division
    # Up to 10,000 is a good balance.
    limit_small = 10000
    sieve = [True] * (limit_small + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit_small**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit_small + 1, i):
                sieve[j] = False
    small_primes = [i for i, is_p in enumerate(sieve) if is_p]
    
    # Generate candidate primes
    # 1. Sieve up to 10^6
    limit_sieve = 10**6
    sieve_p = [True] * (limit_sieve + 1)
    sieve_p[0] = sieve_p[1] = False
    for i in range(2, int(limit_sieve**0.5) + 1):
        if sieve_p[i]:
            for j in range(i*i, limit_sieve + 1, i):
                sieve_p[j] = False
    primes = [i for i, is_p in enumerate(sieve_p) if is_p]
    
    # 2. Add larger primes using Miller-Rabin if time permits
    # We can generate candidates in steps.
    
    for p in primes:
        if time.time() > deadline:
            break
        if p < 3:
            continue
            
        k = 0
        while k < 200:
            n = p * p + k + 1
            if is_p_smooth(n, p, small_primes):
                k += 1
            else:
                break
        
        if k > best_k:
            best_k = k
            best_p = p
            # If we found a very good k, we can print/log but we just return.
            if best_k >= 10:
                # Very unlikely, but if we do, we might want to search nearby?
                # Or just keep going.
                pass

    # If time remains, search larger primes
    if time.time() < deadline - 5:
        # Generate primes > 10^6
        # Step through odd numbers, check primality
        start_check = limit_sieve + 1
        if start_check % 2 == 0:
            start_check += 1
        p = start_check
        # Limit the number of checks to avoid timeout
        # is_prime is fast, is_p_smooth is the bottleneck.
        # Let's check up to 10^7 or until deadline.
        while p < 10**7 and time.time() < deadline:
            if is_prime(p):
                k = 0
                while k < 200:
                    n = p * p + k + 1
                    if is_p_smooth(n, p, small_primes):
                        k += 1
                    else:
                        break
                if k > best_k:
                    best_k = k
                    best_p = p
            p += 2

    return {'p': best_p, 'k': best_k}
