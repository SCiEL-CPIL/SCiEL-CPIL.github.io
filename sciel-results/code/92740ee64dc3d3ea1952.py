import time
import math
import random

def is_prime(n):
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    # Deterministic Miller-Rabin for n < 3,317,044,064,679,887,385,961,981
    # Bases: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37
    # For n < 10^9, bases 2, 3, 5, 7, 11 are enough.
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in [2, 3, 5, 7, 11, 13]:
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

def pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        y = random.randrange(2, n)
        c = random.randrange(1, n)
        m = 128
        g, r, q = 1, 1, 1
        while g == 1:
            x = y
            for _ in range(r):
                y = (y * y + c) % n
            k = 0
            while k < r and g == 1:
                ys = y
                for _ in range(min(m, r - k)):
                    y = (y * y + c) % n
                    q = (q * abs(x - y)) % n
                g = math.gcd(q, n)
                k += m
            r *= 2
        if g == n:
            while True:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
                if g > 1:
                    break
        if g != n:
            return g

def sieve(limit):
    is_prime_flag = [True] * (limit + 1)
    is_prime_flag[0] = is_prime_flag[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime_flag[i]:
            for j in range(i*i, limit + 1, i):
                is_prime_flag[j] = False
    return [i for i, prime in enumerate(is_prime_flag) if prime]

def is_p_smooth(n, p, deadline=None):
    """Check if n has all prime factors <= p. Returns True if p-smooth, False otherwise."""
    if n == 1:
        return True
    if n <= p:
        return True  # n itself is a prime factor <= p (or 1)
    
    # Quick primality test: if n is prime and n > p, it fails
    if is_prime(n):
        return n <= p
    
    # Try to find a factor > p
    # First, try small primes up to min(p, sqrt(n))
    # But we only care about factors > p, so we can skip small ones if possible
    
    # If n is composite, it has a factor <= sqrt(n)
    # If sqrt(n) <= p, then all factors could be <= p, but we need to check
    # Actually, if n has a factor > p, then it must also have a factor < n/p
    # Wait, that's not right. If n = a * b with a <= b, then a <= sqrt(n)
    # If b > p, then a = n/b < n/p
    
    # Strategy: 
    # 1. If n is prime and n > p, return False
    # 2. Try to factor n. If we find a factor f:
    #    - If f > p, return False (since f is a prime factor > p)
    #    - Otherwise, recursively check f and n/f
    # 3. Use Pollard's Rho to find a factor
    
    # Optimization: if n < (p+1)^2, then any composite n must have a factor <= p
    # because if both factors were > p, their product would be > p^2 >= n (since n < (p+1)^2)
    # Wait, that's not quite right. If n < (p+1)^2, then n cannot have two factors both > p
    # So if n is composite, it must have at least one factor <= p
    # But that doesn't mean ALL factors are <= p. We need to check recursively.
    
    # Let's just do recursive factorization with early exit
    
    def _check_p_smooth(m):
        if m == 1:
            return True
        if m <= p:
            return True
        if is_prime(m):
            return m <= p
        
        # Find a factor
        # First try small primes for speed
        # But we only need to find ANY factor, and if that factor > p, we fail
        # Actually, we need to check if ALL prime factors are <= p
        
        # Try Pollard's Rho
        d = pollard_brent(m)
        if d == m:
            # Should not happen if m is composite, but just in case
            return is_prime(m) and m <= p
        
        return _check_p_smooth(d) and _check_p_smooth(m // d)
    
    return _check_p_smooth(n)

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 135s limit, 15s safety margin
    
    # Sieve up to 10^6 first for quick scan
    # Then extend if time permits
    limit = 10**6
    primes = sieve(limit)
    
    best_k = 0
    best_p = 2
    
    for p in primes:
        if time.time() > deadline:
            break
            
        # Check run length for this p
        k = 0
        while k < 200:
            n = p * p + k + 1
            if not is_p_smooth(n, p):
                break
            k += 1
        
        if k > best_k:
            best_k = k
            best_p = p
            
    # If we have time, try larger primes
    if time.time() < deadline - 30:
        # Extend sieve
        limit2 = 10**7
        primes2 = sieve(limit2)
        # Skip primes we already checked
        for p in primes2:
            if p <= limit:
                continue
            if time.time() > deadline:
                break
                
            k = 0
            while k < 200:
                n = p * p + k + 1
                if not is_p_smooth(n, p):
                    break
                k += 1
                
            if k > best_k:
                best_k = k
                best_p = p
                
    return {'p': best_p, 'k': best_k}
