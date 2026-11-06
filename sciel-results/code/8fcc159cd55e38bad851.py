import random
import math
import time

def construct():
    # 1. Sieve of Eratosthenes for small primes
    def get_primes(n):
        sieve = bytearray([True]) * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    # Generate primes up to 10^6 for trial division and sequential search
    primes = get_primes(10**6)
    small_primes = [p for p in primes if p < 10000]

    # 2. Deterministic Miller-Rabin for 64-bit integers
    def is_prime(n):
        if n < 2:
            return False
        if n in (2, 3):
            return True
        if n % 2 == 0:
            return False
        # Write n-1 as d * 2^s
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        
        # Bases sufficient for n < 2^64
        bases = [2, 325, 9375, 28178, 450775, 9780504, 1795265022]
        for a in bases:
            if a % n == 0:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = (x * x) % n
                if x == n - 1:
                    break
            else:
                return False
        return True

    # 3. Pollard's Rho with Brent's cycle detection
    def pollard_rho(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        
        while True:
            y = random.randrange(2, n)
            c = random.randrange(1, n)
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
                r *= 2
            if g == n:
                while True:
                    ys = (ys * ys + c) % n
                    g = math.gcd(abs(x - ys), n)
                    if g > 1:
                        break
            if g != n:
                return g

    # 4. Check if n is p-smooth (all prime factors <= p)
    def is_p_smooth(n, p, small_primes):
        if n == 1:
            return True
        
        # Quick check: if n is prime and n > p, it's not p-smooth
        if n > p and is_prime(n):
            return False

        # Trial division by small primes
        for q in small_primes:
            if q > p:
                break
            if q * q > n:
                break
            if n % q == 0:
                while n % q == 0:
                    n //= q
                if n == 1:
                    return True
                # If remaining n is prime and > p, fail
                if n > p and is_prime(n):
                    return False

        if n == 1:
            return True
        
        if is_prime(n):
            return n <= p
        
        # Use Pollard Rho for composite n
        d = pollard_rho(n)
        return is_p_smooth(d, p, small_primes) and is_p_smooth(n // d, p, small_primes)

    # 5. Search
    best_k = 0
    best_p = 2
    start_time = time.time()
    timeout = 115 # 5s buffer

    # Sequential search through primes up to 10^6
    # This covers a wide range of "cheap" candidates
    for p in primes:
        if time.time() - start_time > timeout:
            break
            
        k = 0
        for i in range(1, 201):
            n = p*p + i
            if not is_p_smooth(n, p, small_primes):
                break
            k += 1
            
        if k > best_k:
            best_k = k
            best_p = p
            if best_k == 200:
                break

    # If time remains, do random search for larger primes
    # Larger primes are more expensive to check, so we do this second
    while time.time() - start_time < timeout:
        # Generate random prime in [10^6, 10^9]
        while True:
            p = random.randrange(10**6, 10**9, 2) | 1
            if is_prime(p):
                break
        
        k = 0
        for i in range(1, 201):
            n = p*p + i
            if not is_p_smooth(n, p, small_primes):
                break
            k += 1
            
        if k > best_k:
            best_k = k
            best_p = p
            if best_k == 200:
                break

    return {'p': best_p, 'k': best_k}
