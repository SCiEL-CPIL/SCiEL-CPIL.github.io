import random
import math

def construct():
    # Deterministic Miller-Rabin for 64-bit integers
    def is_prime(n):
        if n < 2:
            return False
        if n in (2, 3):
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        d, s = n - 1, 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            if a >= n:
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

    # Pollard's Rho with Brent's cycle detection
    def pollard_rho(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        
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
                        q = (q * abs(x - y)) % n
                    g = math.gcd(q, n)
                    k += m
                r *= 2
            if g == n:
                g = 1
                while g == 1:
                    ys = (ys * ys + c) % n
                    g = math.gcd(abs(x - ys), n)
            if g != n:
                return g
            # If g == n, retry with new y, c

    # Check if n is p-smooth
    def is_p_smooth(n, p):
        if n <= p:
            return True
        if is_prime(n):
            return n <= p
        
        d = pollard_rho(n)
        m = n // d
        
        if d <= p:
            # d is p-smooth (factors <= d <= p)
            # Check m
            if is_prime(m):
                return m <= p
            else:
                # m is composite. Since m = n/d and n ~ p^2, m ~ p.
                # If m is composite, its factors are <= sqrt(m) < p (for p > 4).
                # So m is p-smooth.
                return True
        else:
            # d > p
            # m = n/d < p, so m is p-smooth.
            # Check d
            if is_prime(d):
                return d <= p
            else:
                # d is composite. Factors of d are <= sqrt(d) < p (since d ~ p).
                # So d is p-smooth.
                return True

    # Sieve of Eratosthenes
    def sieve(limit):
        is_prime_arr = bytearray(b'\x01') * (limit + 1)
        is_prime_arr[0] = is_prime_arr[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if is_prime_arr[i]:
                is_prime_arr[i*i::i] = bytearray(len(is_prime_arr[i*i::i]))
        return [i for i, prime in enumerate(is_prime_arr) if prime]

    best_k = 0
    best_p = 2

    # Search small primes
    primes = sieve(1000000)
    for p in primes:
        if p > 10**9:
            break
        k = 1
        # Check k from 1 to 200
        while k <= 200:
            n = p * p + k
            if is_p_smooth(n, p):
                k += 1
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p

    # Search random large primes
    # Use remaining time
    import time
    start_time = time.time()
    while time.time() - start_time < 100: # Leave 20s buffer
        p = random.randint(10**6, 10**9)
        if p % 2 == 0:
            p += 1
        # Ensure p is prime
        while not is_prime(p):
            p += 2
            if p > 10**9:
                p = random.randint(10**6, 10**9)
                if p % 2 == 0:
                    p += 1
        
        k = 1
        while k <= 200:
            n = p * p + k
            if is_p_smooth(n, p):
                k += 1
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p

    return {'p': best_p, 'k': best_k}
