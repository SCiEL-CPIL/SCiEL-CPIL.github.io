import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    # Deterministic Miller-Rabin for 64-bit integers
    def is_prime_mr(n):
        if n < 2:
            return False
        if n == 2 or n == 3:
            return True
        if n % 2 == 0:
            return False
        # Witnesses for n < 2^64
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
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

    # Pollard Rho with Brent's cycle detection
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = random.randrange(1, n)
            c = random.randrange(1, n)
            m = random.randrange(1, n)
            g, q, r = 1, 1, 1
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
            if g != n:
                return g

    def factorize(n):
        """Returns a list of prime factors of n"""
        if n == 1:
            return []
        if is_prime_mr(n):
            return [n]
        d = pollard_brent(n)
        return factorize(d) + factorize(n // d)

    # Precompute small primes for trial division
    small_primes = []
    is_prime = [True] * 10000
    for i in range(2, 100):
        if is_prime[i]:
            small_primes.append(i)
            for j in range(i*i, 10000, i):
                is_prime[j] = False
    # Actually, let's just generate primes up to 10000 properly
    small_primes = []
    limit = 10000
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    for i in range(2, limit + 1):
        if sieve[i]:
            small_primes.append(i)

    def has_large_factor(n, p):
        """Returns True if n has a prime factor > p"""
        if n == 1:
            return False
        # Trial division by small primes
        temp_n = n
        for q in small_primes:
            if q > p:
                break
            if temp_n % q == 0:
                while temp_n % q == 0:
                    temp_n //= q
                if temp_n == 1:
                    return False
        if temp_n == 1:
            return False
        # Check if remaining part is prime
        if is_prime_mr(temp_n):
            return temp_n > p
        # If composite, find a factor
        # If we can find a factor d such that d > p, return True
        # If d <= p, we need to check the rest
        # Since we've removed small factors, any factor of temp_n is > 10000 (or > p if p < 10000)
        # If p < 10000, and temp_n > 1, then temp_n has a factor > p (since all factors <= p were removed).
        # Wait, if p < 10000, we iterated through all q <= p. So all factors <= p are removed.
        # So if temp_n > 1, it has a factor > p.
        if p < limit:
            return True
        
        # If p >= 10000, we need to factor temp_n further
        # Use Pollard Rho
        factors = factorize(temp_n)
        for f in factors:
            if f > p:
                return True
        return False

    best_k = 0
    best_p = 2

    # Search primes
    # We'll generate primes on the fly or use a sieve up to a certain limit
    # Let's search up to 10^6 first, then maybe random larger ones if time permits
    
    # Sieve up to 10^6
    max_p_search = 1000000
    sieve_large = bytearray(b'\x01') * (max_p_search + 1)
    sieve_large[0] = sieve_large[1] = 0
    for i in range(2, int(max_p_search**0.5) + 1):
        if sieve_large[i]:
            for j in range(i*i, max_p_search + 1, i):
                sieve_large[j] = 0
    
    primes_to_check = [i for i in range(2, max_p_search + 1) if sieve_large[i]]
    
    for p in primes_to_check:
        if time.time() > deadline:
            break
        if p <= best_p and best_k == 200: # Optimization
            continue
            
        k = 0
        for i in range(1, 201):
            n = p * p + i
            if has_large_factor(n, p):
                break
            k += 1
            if k > best_k:
                best_k = k
                best_p = p
                if best_k == 200:
                    # We can't do better than 200
                    # But maybe a smaller p? The score is k. So 200 is max.
                    # We can break early if we want to save time, but let's finish this p to be sure?
                    # Actually, if k=200, it's the max possible. We can return immediately?
                    # No, we should ensure we don't miss a case where k=200 is found for a smaller p?
                    # The score is just k. So any p with k=200 is equally good.
                    # So we can return immediately upon finding k=200.
                    return {'p': p, 'k': 200}
                    
    # If we haven't found 200, we might try some random larger primes
    # But factoring numbers ~ 10^18 is slow.
    # Let's stick to the result found.
    
    return {'p': best_p, 'k': best_k}
