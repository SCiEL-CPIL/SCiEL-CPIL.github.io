import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s to be safe
    
    # Deterministic Miller-Rabin for n < 2^64
    def is_prime(n):
        if n < 2:
            return False
        if n in (2, 3):
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        # Witnesses for n < 2^64
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        for a in witnesses:
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

    # Pollard's Rho with Brent's cycle detection
    def pollard_rho(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = 2
            c = 1
            m = 128
            g = r = q = 1
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
                r += 1
            if g == n:
                g = 1
                while g == 1:
                    ys = (ys * ys + c) % n
                    g = math.gcd(abs(x - ys), n)
            if g != n:
                return g
            c += 1

    def factorize(n):
        factors = []
        while n > 1:
            if is_prime(n):
                factors.append(n)
                break
            d = pollard_rho(n)
            factors.append(d)
            n //= d
        return factors

    def max_prime_factor(n):
        if n == 1:
            return 1
        if is_prime(n):
            return n
        factors = factorize(n)
        return max(factors)

    # Sieve of Eratosthenes for primes up to limit
    def sieve(limit):
        is_prime_arr = bytearray(b'\x01') * (limit + 1)
        is_prime_arr[0] = is_prime_arr[1] = 0
        for i in range(2, int(limit ** 0.5) + 1):
            if is_prime_arr[i]:
                is_prime_arr[i*i::i] = bytearray(len(is_prime_arr[i*i::i]))
        return [i for i, val in enumerate(is_prime_arr) if val]

    # Search for best p
    # We search primes p. 
    # Start with a reasonable limit, say 10^6, and expand if time permits.
    # Or just generate primes on the fly up to a high limit.
    
    best_p = 41
    best_k = 3 # Known from manual check: p=41, k=3
    
    # Let's search primes up to 10^7
    # Generating primes up to 10^7 takes a bit of memory/time but is okay.
    # 10^7 bytes is 10MB.
    primes = sieve(10_000_000)
    
    for p in primes:
        if time.time() > deadline:
            break
        if p <= best_p and best_k >= 200: # If we found a very good one, maybe skip small? No, just continue.
            pass
        
        k = 0
        # Check p^2 + 1 ... p^2 + k
        # Optimization: if p is large, p^2 is large.
        # We check sequentially.
        while k < 200:
            n = p * p + k + 1
            # Quick check: if n is prime, it's > p (since n > p^2 > p for p>=2)
            # So if n is prime, it fails.
            if is_prime(n):
                break
            # If n is composite, check max factor
            # We can check if any factor > p
            # Instead of full factorization, we can try to find a factor > p?
            # Or just find max factor.
            # Pollard Rho is fast.
            try:
                mf = max_prime_factor(n)
                if mf > p:
                    break
                k += 1
            except Exception:
                break
        
        if k > best_k:
            best_k = k
            best_p = p

    return {'p': best_p, 'k': best_k}
