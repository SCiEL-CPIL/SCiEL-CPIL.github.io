import time
import math
import random

def construct():
    start_time = time.time()
    deadline = 115  # seconds, safety margin
    
    # Deterministic Miller-Rabin for n < 3.3 * 10^24
    # Witnesses: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37
    def is_prime(n):
        if n < 2:
            return False
        # Small primes
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes:
            if n == p:
                return True
            if n % p == 0:
                return False
        # Write n-1 as d * 2^s
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        # Test witnesses
        for a in small_primes:
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
        if n % 5 == 0:
            return 5
        while True:
            y = random.randint(1, n - 1)
            c = random.randint(1, n - 1)
            m = 128  # batch size
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
                    if g != 1:
                        break
            if g != n:
                return g

    # Pre-compute primes up to a reasonable limit for trial division
    def sieve(limit):
        is_p = bytearray([1]) * (limit + 1)
        is_p[0] = is_p[1] = 0
        for i in range(2, int(limit ** 0.5) + 1):
            if is_p[i]:
                for j in range(i * i, limit + 1, i):
                    is_p[j] = 0
        return [i for i in range(limit + 1) if is_p[i]]

    # Use a larger set of primes for trial division
    trial_primes = sieve(100000)
    
    def is_p_smooth(n, p):
        if n <= p:
            return True
        if is_prime(n):
            return False
        
        # Trial division by pre-computed primes
        for q in trial_primes:
            if q > p:
                break
            if n % q == 0:
                while n % q == 0:
                    n //= q
                    if n <= p:
                        return True
                if n == 1:
                    return True
        
        if n == 1:
            return True
        if n <= p:
            return True
        
        # If n is prime here, it must be > p
        if is_prime(n):
            return False
        
        # Find a factor using Pollard Rho
        d = pollard_rho(n)
        if d == n:
            return False
        
        # Recurse
        return is_p_smooth(d, p) and is_p_smooth(n // d, p)

    best_k = 0
    best_p = 2
    
    # Initial small check
    for p in [2, 3, 5, 7]:
        k = 0
        while k < 200:
            n = p * p + k + 1
            if is_p_smooth(n, p):
                k += 1
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p

    # Search larger primes with adaptive limits
    search_limits = [10**4, 10**5, 10**6, 10**7, 10**8, 10**9]
    prev_limit = 10
    for limit in search_limits:
        if time.time() - start_time > deadline - 15:
            break
        
        primes = sieve(limit)
        for p in primes:
            if p <= prev_limit:
                continue
            if time.time() - start_time > deadline - 5:
                break
            
            k = 0
            while k < 200:
                n = p * p + k + 1
                if is_p_smooth(n, p):
                    k += 1
                else:
                    break
            
            if k > best_k:
                best_k = k
                best_p = p
        
        prev_limit = limit

    return {'p': best_p, 'k': best_k}

if __name__ == "__main__":
    res = construct()
    print(res)
