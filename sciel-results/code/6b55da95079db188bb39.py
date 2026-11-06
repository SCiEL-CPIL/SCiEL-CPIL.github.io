import math
import time
import random
from functools import lru_cache

def construct():
    start_time = time.time()
    TIME_LIMIT = 115  # Safety margin for 120s limit

    def is_prime(n):
        if n < 2:
            return False
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes:
            if n % p == 0:
                return n == p
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in witnesses:
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
                if g > 1:
                    break
        return g

    def has_large_factor(n, p):
        # Trial division by small primes
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for q in small_primes:
            if n % q == 0:
                if q > p:
                    return True
                while n % q == 0:
                    n //= q
                if n == 1:
                    return False
        
        if n == 1:
            return False
            
        if is_prime(n):
            return n > p
            
        # Find a factor
        f = pollard_brent(n)
        if f > p:
            return True
        n //= f
        return has_large_factor(n, p)

    def generate_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, prime in enumerate(sieve) if prime]

    # Pre-compute primes up to 10^4 for fast filtering
    SMALL_PRIME_LIMIT = 10000
    small_primes = generate_primes(SMALL_PRIME_LIMIT)
    
    # Pre-compute primes up to 10^6 for trial division in exact check
    # We will use these to speed up the exact check
    MEDIUM_PRIME_LIMIT = 100000
    medium_primes = generate_primes(MEDIUM_PRIME_LIMIT)

    best_p = 7
    best_k = 1
    
    # Search strategy:
    # 1. Check small primes up to 10^5 thoroughly.
    # 2. Check random primes up to 10^9 with a fast filter.
    
    # Phase 1: Small primes
    primes_up_to_100k = generate_primes(100000)
    for p in primes_up_to_100k:
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        k = 0
        base = p * p
        while k < 200:
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            n = base + k + 1
            if has_large_factor(n, p):
                break
            k += 1
        if k > best_k:
            best_k = k
            best_p = p

    # Phase 2: Large primes with fast filter
    # We iterate through odd numbers, check primality, then filter.
    # To make this efficient, we only do the full check if the fast filter passes.
    
    # Fast filter: Check if p^2 + 1 ... p^2 + 20 are 10^4-smooth
    # If a number is not 10^4-smooth, it has a prime factor > 10^4.
    # If that factor is <= p, it might still be p-smooth.
    # But if the factor is > p, it's definitely not p-smooth.
    # So the filter is: for each n, remove all factors <= 10^4.
    # If remainder R > 1, check if R is prime. If prime and R > p, fail.
    # If R is composite, it's hard to say without full factorization.
    # So the filter is weak.
    
    # Better filter: 
    # If n has a prime factor q > p, then n is not p-smooth.
    # We can check divisibility by primes in (10^4, p] ? No, too many.
    
    # Let's rely on the fact that small p yields small k, and large p might yield larger k?
    # Actually, the probability of smoothness is ~0.3.
    # The expected run length is 1/(1-0.3) = 1.4.
    # So k=10 is rare.
    
    # We will just do a random search for large primes.
    random.seed(42)
    
    for _ in range(10000): # Try 10000 random primes
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        # Generate a random odd number up to 10^9
        p_candidate = random.randrange(100000, 10**9, 2)
        if not is_prime(p_candidate):
            continue
        
        p = p_candidate
        k = 0
        base = p * p
        # Use medium primes for trial division to speed up
        while k < 200:
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            n = base + k + 1
            if has_large_factor(n, p):
                break
            k += 1
        if k > best_k:
            best_k = k
            best_p = p

    return {'p': best_p, 'k': best_k}

if __name__ == '__main__':
    construct()
