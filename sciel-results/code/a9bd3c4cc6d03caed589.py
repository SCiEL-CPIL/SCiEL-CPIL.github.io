import math
import time
import random

def construct():
    start_time = time.time()
    TIME_LIMIT = 145  # safety margin from 150s kill

    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        # Deterministic Miller-Rabin for n < 3,317,044,064,679,887,385,961,981
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = x * x % n
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
        if n % 5 == 0:
            return 5
        while True:
            y = random.randint(1, n - 1)
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
                g = 1
                while g == 1:
                    ys = (ys * ys + c) % n
                    g = math.gcd(abs(x - ys), n)
            if g != n:
                return g

    def largest_prime_factor(n, p_limit):
        """Return the largest prime factor of n, but short-circuit if we find one > p_limit."""
        if n <= 1:
            return 1
        max_pf = 1
        # Trial divide by small primes first
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
            if p > p_limit:
                break
            while n % p == 0:
                max_pf = max(max_pf, p)
                n //= p
        if n == 1:
            return max_pf
        if is_prime(n):
            return max(max_pf, n)
        # Use Pollard-Brent for remaining composite
        stack = [n]
        while stack:
            x = stack.pop()
            if x == 1:
                continue
            if is_prime(x):
                if x > p_limit:
                    return x  # short-circuit: found factor > p_limit
                max_pf = max(max_pf, x)
            else:
                d = pollard_brent(x)
                if d == x:
                    stack.append(x)
                else:
                    stack.append(d)
                    stack.append(x // d)
        return max_pf

    def check_run(p, max_k=200):
        """Check how many consecutive integers p^2+1 ... p^2+k have all prime factors <= p."""
        k = 0
        for i in range(1, max_k + 1):
            if time.time() - start_time > TIME_LIMIT - 3:
                break
            n = p * p + i
            lpf = largest_prime_factor(n, p)
            if lpf <= p:
                k += 1
            else:
                break
        return k

    best_k = 0
    best_p = 2

    # Precompute small primes up to 10000 for quick primality and trial division
    small_primes = []
    for x in range(2, 10000):
        if all(x % q for q in small_primes if q * q <= x):
            small_primes.append(x)

    def is_prime_fast(n):
        """Fast primality test using precomputed small primes then Miller-Rabin."""
        if n < 2:
            return False
        for p in small_primes:
            if p * p > n:
                return True
            if n % p == 0:
                return n == p
        return is_prime(n)

    # Phase 1: Check small primes (quick, likely short runs)
    for p in small_primes:
        if time.time() - start_time > TIME_LIMIT - 60:
            break
        k = check_run(p)
        if k > best_k:
            best_k = k
            best_p = p

    # Phase 2: Search medium primes (10^4 to 10^6)
    p = 10000
    while p < 10**6:
        if time.time() - start_time > TIME_LIMIT - 30:
            break
        if is_prime_fast(p):
            k = check_run(p)
            if k > best_k:
                best_k = k
                best_p = p
        p += 2

    # Phase 3: Random search in large prime range (10^6 to 10^9)
    random.seed(12345)
    attempts = 0
    while time.time() - start_time < TIME_LIMIT - 5:
        attempts += 1
        # Bias toward larger primes where long runs are more likely
        if attempts < 50:
            p = random.randint(10**6, 10**7)
        elif attempts < 150:
            p = random.randint(10**7, 10**8)
        else:
            p = random.randint(10**8, 10**9)
        if p % 2 == 0:
            p += 1
        # Find next prime
        while not is_prime_fast(p) and p < 10**9:
            p += 2
        if p >= 10**9:
            continue
        k = check_run(p)
        if k > best_k:
            best_k = k
            best_p = p

    # Fallback
    if best_k < 1:
        best_p = 2
        best_k = 1

    return {'p': best_p, 'k': best_k}
