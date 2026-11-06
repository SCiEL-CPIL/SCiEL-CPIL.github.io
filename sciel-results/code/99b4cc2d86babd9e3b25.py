import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110

    # Miller-Rabin
    def is_prime(n):
        if n < 2: return False
        if n in (2, 3): return True
        if n % 2 == 0: return False
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if a >= n: continue
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
        if n % 2 == 0: return 2
        if n % 3 == 0: return 3
        while True:
            y = random.randrange(1, n)
            c = random.randrange(1, n)
            m = random.randrange(1, n)
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

    def factorize(n, factors):
        if n == 1: return
        if is_prime(n):
            factors.append(n)
            return
        d = pollard_brent(n)
        factorize(d, factors)
        factorize(n // d, factors)

    # Small primes for trial division
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    def is_p_smooth(N, p):
        R = N
        for q in small_primes:
            if q > p: break
            while R % q == 0:
                R //= q
        if R == 1:
            return True
        # If R is prime, check if R <= p
        if is_prime(R):
            return R <= p
        # If R is composite, factor it
        factors = []
        factorize(R, factors)
        for f in factors:
            if f > p:
                return False
        return True

    best_k = 0
    best_p = 2
    
    # Generate primes up to 10^7 for search
    # Sieve
    SIEVE_LIMIT = 10_000_000
    is_prime_sieve = [True] * (SIEVE_LIMIT + 1)
    is_prime_sieve[0] = is_prime_sieve[1] = False
    for i in range(2, int(SIEVE_LIMIT**0.5) + 1):
        if is_prime_sieve[i]:
            for j in range(i*i, SIEVE_LIMIT + 1, i):
                is_prime_sieve[j] = False
    primes = [i for i in range(2, SIEVE_LIMIT + 1) if is_prime_sieve[i]]
    
    for p in primes:
        if time.time() > deadline:
            break
            
        k = 0
        p_sq = p * p
        for j in range(1, 201):
            if is_p_smooth(p_sq + j, p):
                k += 1
            else:
                break
                
        if k > best_k:
            best_k = k
            best_p = p

    return {'p': best_p, 'k': best_k}
