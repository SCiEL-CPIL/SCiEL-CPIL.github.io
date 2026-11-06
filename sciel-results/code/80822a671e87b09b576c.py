import math
import time

def construct():
    deadline = time.time() + 110
    
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
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
            y = 2
            c = 1
            m = 128
            g = 1
            r = 1
            q = 1
            while g == 1:
                x = y
                for _ in range(r):
                    y = (y * y + c) % n
                k = 0
                while k < r and g == 1:
                    ys = x
                    for _ in range(min(m, r - k)):
                        x = (x * x + c) % n
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
            c += 1
    
    def factorize(n):
        if n <= 1:
            return []
        factors = []
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        for p in small_primes:
            while n % p == 0:
                factors.append(p)
                n //= p
        if n == 1:
            return factors
        if is_prime(n):
            factors.append(n)
            return factors
        stack = [n]
        while stack:
            num = stack.pop()
            if num == 1:
                continue
            if is_prime(num):
                factors.append(num)
            else:
                d = pollard_brent(num)
                stack.append(d)
                stack.append(num // d)
        return factors
    
    def is_p_smooth(n, p):
        if n <= 1:
            return True
        # If n <= p, then n is trivially p-smooth (its only prime factor is <= n <= p)
        if n <= p:
            return True
        # Factorize n and check all prime factors <= p
        for f in factorize(n):
            if f > p:
                return False
        return True
    
    def sieve(limit):
        is_p = [True] * (limit + 1)
        is_p[0] = is_p[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_p[i]:
                for j in range(i*i, limit + 1, i):
                    is_p[j] = False
        return [i for i in range(2, limit + 1) if is_p[i]]
    
    best_k = 1
    best_p = 2
    
    # Search small primes first
    primes_small = sieve(10**6)
    
    for p in primes_small:
        if time.time() > deadline:
            break
        k = 0
        p2 = p * p
        while k < 200:
            if time.time() > deadline:
                break
            n = p2 + k + 1
            if is_p_smooth(n, p):
                k += 1
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p
            if best_k >= 200:
                break
    
    # Try some larger primes if time permits
    if time.time() < deadline - 10:
        # Test random-ish large primes
        import random
        random.seed(42)
        tested = 0
        for _ in range(2000):
            if time.time() > deadline - 5:
                break
            n = random.randint(10**6, 10**9)
            if n % 2 == 0:
                n += 1
            # Find next prime
            while not is_prime(n):
                n += 2
                if n > 10**9:
                    break
            if n > 10**9:
                break
            tested += 1
            k = 0
            p2 = n * n
            while k < 200:
                if time.time() > deadline - 5:
                    break
                val = p2 + k + 1
                if is_p_smooth(val, n):
                    k += 1
                else:
                    break
            if k > best_k:
                best_k = k
                best_p = n
                if best_k >= 200:
                    break
    
    return {'p': best_p, 'k': best_k}
