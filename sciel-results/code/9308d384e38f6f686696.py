import time
import math
import random

def _is_prime_mr(n):
    """Deterministic Miller-Rabin for n < 3.3e24."""
    if n < 2:
        return False
    if n in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        return True
    if n % 2 == 0:
        return False
    
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
        
    # Bases sufficient for n < 3.3e24
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
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

def _pollard_brent(n):
    """Pollard's Brent cycle detection for finding a factor of n."""
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    
    y, c, m = random.randint(1, n - 1), random.randint(1, n - 1), random.randint(1, n - 1)
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
                
    return g

def _sieve(limit):
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    return [i for i, prime in enumerate(is_prime) if prime]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Precompute small primes for trial division
    small_primes = _sieve(100000) # 10^5 primes
    
    def is_p_smooth(n, p):
        """Check if n is p-smooth using small_primes and Pollard-Brent."""
        original_n = n
        # Trial division by small primes
        for q in small_primes:
            if q > p or q * q > n:
                break
            while n % q == 0:
                n //= q
        
        if n == 1:
            return True
            
        if _is_prime_mr(n):
            return n <= p
            
        # n is composite and > 1
        # Find a factor
        try:
            f = _pollard_brent(n)
        except Exception:
            return False # Fallback, assume not smooth to be safe? Or try again?
        
        return is_p_smooth(f, p) and is_p_smooth(n // f, p)

    best_k = 0
    best_p = 0
    
    # Search small primes first
    # Generate primes up to 10^6 for the main loop
    search_primes = _sieve(1000000)
    
    for p in search_primes:
        if time.time() > deadline:
            break
            
        k = 0
        base = p * p
        while k < 200:
            if time.time() > deadline:
                break
            n = base + k + 1
            if is_p_smooth(n, p):
                k += 1
            else:
                break
                
        if k > best_k:
            best_k = k
            best_p = p
            # If we find a very large k, we might stop early?
            # But we want to maximize.
            
    # If we didn't find a high k, or just to be safe, do some random sampling of larger primes
    # Only if we have time
    if time.time() < deadline - 10:
        # Try to find a prime with larger k
        # Random search
        attempts = 0
        while time.time() < deadline and attempts < 1000:
            attempts += 1
            # Generate random odd number in [10^6, 10^9]
            p_candidate = random.randint(10**6, 10**9)
            if p_candidate % 2 == 0:
                p_candidate += 1
                
            if _is_prime_mr(p_candidate):
                k = 0
                base = p_candidate * p_candidate
                while k < 200:
                    if time.time() > deadline:
                        break
                    n = base + k + 1
                    if is_p_smooth(n, p_candidate):
                        k += 1
                    else:
                        break
                        
                if k > best_k:
                    best_k = k
                    best_p = p_candidate

    return {'p': best_p, 'k': best_k}
