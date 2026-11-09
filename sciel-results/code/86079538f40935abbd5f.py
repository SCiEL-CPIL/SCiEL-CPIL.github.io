import time
import math
import random

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin

    # Deterministic Miller-Rabin for n < 3.3e24
    def is_prime_mr(n):
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

    # Pollard's Brent cycle detection
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        if n % 5 == 0:
            return 5
        while True:
            y = random.randint(2, n - 1)
            c = random.randint(1, n - 1)
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

    def sieve_primes(limit):
        is_p = bytearray(b'\x01') * (limit + 1)
        is_p[0] = is_p[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if is_p[i]:
                for j in range(i*i, limit+1, i):
                    is_p[j] = 0
        return [i for i in range(2, limit+1) if is_p[i]]

    # Precompute small primes for batch division
    SMALL_PRIME_LIMIT = 100000
    small_primes = sieve_primes(SMALL_PRIME_LIMIT)
    
    # Primes for search up to 10^6
    search_primes = sieve_primes(1000000)

    def check_run_batch(p, k_max=200):
        """
        Check how many consecutive integers p^2+1, ..., p^2+k_max are p-smooth.
        """
        if time.time() - start_time > DEADLINE:
            return 0
        
        base = p * p
        k = 0
        for i in range(1, k_max + 1):
            if time.time() - start_time > DEADLINE:
                break
            
            n = base + i
            t = n
            
            # Divide by small primes
            for q in small_primes:
                if q > p:
                    break
                if q * q > t:
                    break
                while t % q == 0:
                    t //= q
            
            if t == 1:
                k = i
                continue
            
            # Check if t is prime
            if is_prime_mr(t):
                if t <= p:
                    k = i
                else:
                    break
            else:
                # t is composite
                # Factor t and check if all factors <= p
                # Since we divided by q <= p (up to limit), if p < limit, t has no factors <= p.
                # If p >= limit, t might have factors <= p that are > limit.
                
                # If p < SMALL_PRIME_LIMIT, we have divided by all q <= p.
                # So if t > 1, it has a factor > p. Fail.
                if p < SMALL_PRIME_LIMIT:
                    break
                
                # If p >= SMALL_PRIME_LIMIT, we only divided by q <= SMALL_PRIME_LIMIT.
                # t might have prime factors in (SMALL_PRIME_LIMIT, p].
                # We need to factor t and check.
                try:
                    factors = []
                    def factor_rec(m):
                        if m == 1:
                            return
                        if is_prime_mr(m):
                            factors.append(m)
                            return
                        f = pollard_brent(m)
                        factor_rec(f)
                        factor_rec(m // f)
                    
                    factor_rec(t)
                    if all(f <= p for f in factors):
                        k = i
                    else:
                        break
                except Exception:
                    break
        
        return k

    best_k = 0
    best_p = 2

    # Phase 1: Search primes up to 10^6
    for p in search_primes:
        if time.time() - start_time > DEADLINE:
            break
        k = check_run_batch(p, 200)
        if k > best_k:
            best_k = k
            best_p = p
            if best_k >= 200:
                break

    # Phase 2: Search larger primes
    # Random sampling in [10^6, 10^9]
    if best_k < 200:
        # Prioritize larger p
        for _ in range(5000):
            if time.time() - start_time > DEADLINE:
                break
            
            # Log-uniform sampling
            log_val = 6 + random.random() * 3
            p = int(10 ** log_val)
            if p % 2 == 0:
                p += 1
            while not is_prime_mr(p):
                p += 2
                if p > 10**9:
                    break
            if p > 10**9:
                continue
                
            k = check_run_batch(p, 200)
            if k > best_k:
                best_k = k
                best_p = p
                if best_k >= 200:
                    break

    return {'p': best_p, 'k': best_k}
