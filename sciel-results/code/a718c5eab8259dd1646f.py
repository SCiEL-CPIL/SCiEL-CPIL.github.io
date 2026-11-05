import math
import time
import random

def construct():
    deadline = time.time() + 113.0  # safety margin before 120s

    def sieve(limit):
        is_p = bytearray(b'\x01') * (limit + 1)
        is_p[0:2] = b'\x00\x00'
        for i in range(2, int(limit**0.5) + 1):
            if is_p[i]:
                is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
        return [i for i in range(2, limit + 1) if is_p[i]]

    SIEVE_LIMIT = 1000000
    primes = sieve(SIEVE_LIMIT)
    prime_set = set(primes)
    
    def is_prime_mr(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        small_primes_mr = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for sp in small_primes_mr:
            if n == sp:
                return True
            if n % sp == 0:
                return False
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
        """Pollard's Brent cycle detection factorization."""
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        y, c, m = random.randint(1, n - 1), random.randint(1, n - 1), 128
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

    def is_p_smooth(n, p, primes_list):
        """Check if n is p-smooth using primes_list (primes <= min(p, SIEVE_LIMIT))."""
        if n <= 1:
            return True
        
        # Sieve with primes in primes_list
        # primes_list contains primes up to SIEVE_LIMIT.
        # We only use those <= p.
        for q in primes_list:
            if q > p:
                break
            if q * q > n:
                break
            while n % q == 0:
                n //= q
            if n == 1:
                return True
        
        if n == 1:
            return True
        
        # If p < SIEVE_LIMIT, we have sieved all possible small factors.
        # If n > 1, it must be prime (since any composite factor would have a prime factor <= sqrt(n) <= p? 
        # Wait, if p < SIEVE_LIMIT, we sieved all q <= p.
        # If n > 1, n has no prime factors <= p.
        # So n is either prime > p, or product of primes > p.
        # If n is prime > p, not smooth.
        # If n is composite, it has factors > p, not smooth.
        # So if p < SIEVE_LIMIT and n > 1, return False.
        if p < SIEVE_LIMIT:
            return False
            
        # If p >= SIEVE_LIMIT, we have sieved all primes <= SIEVE_LIMIT.
        # n has no prime factors <= SIEVE_LIMIT.
        # We need to check if n has any prime factor > p.
        # If n <= p, then n is prime (since no factors <= SIEVE_LIMIT < p? No, SIEVE_LIMIT <= p).
        # If n <= p, and n > 1, and no factors <= SIEVE_LIMIT (which is <= p), then n must be prime?
        # Yes, if n is composite, it has a factor <= sqrt(n) <= sqrt(p) < p.
        # But we sieved all factors <= SIEVE_LIMIT.
        # If SIEVE_LIMIT >= sqrt(n), then we would have found a factor if n was composite.
        # So if n <= SIEVE_LIMIT^2, and no factors found, n is prime.
        
        # Check if n is prime
        if is_prime_mr(n):
            return n <= p
        
        # n is composite.
        # Factor n.
        # n has factors > SIEVE_LIMIT.
        try:
            factor = pollard_brent(n)
            # Recursively check
            if not is_p_smooth(factor, p, primes_list):
                return False
            if not is_p_smooth(n // factor, p, primes_list):
                return False
            return True
        except Exception:
            return False

    def check_run(p):
        """Check how many consecutive numbers p^2+1 .. p^2+k are p-smooth."""
        k = 0
        p_sq = p * p
        for k_val in range(1, 201):
            v = p_sq + k_val
            if is_p_smooth(v, p, primes):
                k = k_val
            else:
                break
        return k

    best_k = 0
    best_p = 2

    # Check p = 2
    k = check_run(2)
    if k > best_k:
        best_k = k
        best_p = 2

    # Systematic search over primes
    # We can check primes up to SIEVE_LIMIT efficiently.
    for p in primes:
        if time.time() > deadline:
            break
        k = check_run(p)
        if k > best_k:
            best_k = k
            best_p = p
        if best_k >= 190:
            break

    # Try larger primes beyond SIEVE_LIMIT
    # We can't sieve all primes, but we can sample or search intervals.
    # However, the hybrid is_p_smooth works for p > SIEVE_LIMIT too.
    # Let's try some larger primes.
    random.seed(42)
    checked = set(primes)
    
    # Search in descending order from large values, but sparse
    # Since p^2 is large, Pollard-Brent might be slower, but rare.
    for _ in range(5000):
        if time.time() > deadline:
            break
        candidate = random.randint(1000000, 100000000)
        if candidate in checked:
            continue
        checked.add(candidate)
        if not is_prime_mr(candidate):
            continue
        p = candidate
        k = check_run(p)
        if k > best_k:
            best_k = k
            best_p = p

    return {'p': best_p, 'k': best_k}
