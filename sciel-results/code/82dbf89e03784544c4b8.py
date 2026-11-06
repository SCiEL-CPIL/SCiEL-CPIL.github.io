import math
import time
import random

def construct():
    start_time = time.time()
    TIME_LIMIT = 110  # Leave 10s margin for 120s limit

    # Pre-compute small primes for trial division
    LIMIT_SMALL = 10**5
    small_primes = []
    is_prime_small = [True] * (LIMIT_SMALL + 1)
    is_prime_small[0] = is_prime_small[1] = False
    for i in range(2, int(LIMIT_SMALL**0.5) + 1):
        if is_prime_small[i]:
            for j in range(i*i, LIMIT_SMALL + 1, i):
                is_prime_small[j] = False
    for i in range(2, LIMIT_SMALL + 1):
        if is_prime_small[i]:
            small_primes.append(i)

    def is_prime(n):
        if n < 2: return False
        for p in small_primes:
            if p * p > n: break
            if n % p == 0:
                return n == p
        # Miller-Rabin for larger n
        # Deterministic for n < 3,317,044,064,679,887,385,961,981
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in witnesses:
            if a >= n: continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1: continue
            for _ in range(s - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True

    def pollard_brent(n):
        if n % 2 == 0: return 2
        if n % 3 == 0: return 3
        if n % 5 == 0: return 5
        y, c, m = random.randrange(1, n), random.randrange(1, n), random.randrange(1, n)
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
                    q = q * abs(x - y) % n
                g = math.gcd(q, n)
                k += m
            r *= 2
        if g == n:
            while True:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
                if g > 1: break
        return g

    def is_p_smooth(n, p):
        if n == 1:
            return True
        if is_prime(n):
            return n <= p
        
        # Trial division by small primes
        # We only need to divide by primes <= p
        # But we only have small_primes up to 10^5
        # If p > 10^5, we divide by all small_primes
        # If p < 10^5, we divide by small_primes <= p
        
        remaining = n
        for d in small_primes:
            if d > p:
                break
            if d * d > remaining:
                break
            while remaining % d == 0:
                remaining //= d
            if remaining == 1:
                return True
        
        if remaining == 1:
            return True
            
        # Now remaining is not divisible by any small prime <= p
        # If remaining is prime, check if <= p
        if is_prime(remaining):
            return remaining <= p
            
        # If remaining is composite, it must have a prime factor > 10^5 (or > p if p < 10^5)
        # Find a factor
        f = pollard_brent(remaining)
        # Check if f is p-smooth and remaining/f is p-smooth
        # Optimization: if f > p and is_prime(f), return False
        if f > p and is_prime(f):
            return False
        if f == remaining:
            # Pollard failed to find proper factor? Should not happen if composite
            return is_prime(remaining) and remaining <= p
            
        return is_p_smooth(f, p) and is_p_smooth(remaining // f, p)

    best_p = 0
    best_k = 0
    
    # Search strategy
    # Try different ranges of p
    # We can try to search primes in [10^4, 10^6], [10^6, 10^7], etc.
    # Or just random sampling with a good factorization.
    
    # Let's try random sampling but with a bias towards smaller p?
    # Or just iterate?
    # Iterating from 10^4 to 10^7 is 10^7 primes, too many.
    # Random sampling is better.
    
    # To ensure we find a good one, we can try multiple random points.
    
    for _ in range(10000): # Max iterations, will be cut by time
        if time.time() - start_time > TIME_LIMIT:
            break
        
        # Pick random base
        # Bias towards smaller p? Smaller p might have easier smoothness?
        # Actually, larger p makes the condition "factor <= p" easier to satisfy.
        # But p^2 is larger, so factors are larger.
        # Let's try uniform in [10^4, 10^7] first, then expand.
        
        # Let's try a mix
        if random.random() < 0.5:
            base = random.randint(10**4, 10**6)
        else:
            base = random.randint(10**6, 10**8)
            
        p = base
        # Find next prime
        while not is_prime(p):
            p += 1
            if p > 10**9:
                break
        
        if p > 10**9:
            continue
            
        k = 0
        base_sq = p * p
        while k < 200:
            if time.time() - start_time > TIME_LIMIT:
                break
            n = base_sq + k + 1
            if is_p_smooth(n, p):
                k += 1
            else:
                break
        
        if k > best_k:
            best_k = k
            best_p = p
            # If we find a high k, we can try to refine around it?
            # Or just keep searching.

    return {'p': best_p, 'k': best_k}
