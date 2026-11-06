import time
import math
import random

def construct():
    start_time = time.time()
    deadline = 115  # seconds, safety margin

    # Deterministic Miller-Rabin for n < 3.3 * 10^24
    def is_prime(n):
        if n < 2:
            return False
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes:
            if n == p:
                return True
            if n % p == 0:
                return False
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
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
                    if g != 1:
                        break
            if g != n:
                return g

    # Precompute small primes for trial division
    def sieve_small(limit):
        is_p = bytearray(b'\x01') * (limit + 1)
        is_p[0] = is_p[1] = 0
        for i in range(2, int(limit ** 0.5) + 1):
            if is_p[i]:
                for j in range(i * i, limit + 1, i):
                    is_p[j] = 0
        return [i for i in range(limit + 1) if is_p[i]]

    small_primes = sieve_small(10000)
    small_primes_set = set(small_primes)

    def is_p_smooth(n, p):
        """Check if n has all prime factors <= p"""
        if n <= p:
            return True
        if n == 1:
            return True
        
        # Trial division by small primes
        for q in small_primes:
            if q > p:
                break
            if q * q > n:
                break
            while n % q == 0:
                n //= q
                if n <= p:
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
        
        # Recurse on both parts
        return is_p_smooth(d, p) and is_p_smooth(n // d, p)

    def check_run(p, max_k=200):
        """Check how many consecutive integers p^2+1...p^2+k are p-smooth"""
        k = 0
        p2 = p * p
        while k < max_k:
            n = p2 + k + 1
            if not is_p_smooth(n, p):
                break
            k += 1
            if time.time() - start_time > deadline - 5:
                break
        return k

    # Sieve for searching primes
    def sieve(limit):
        is_p = bytearray(b'\x01') * (limit + 1)
        is_p[0] = is_p[1] = 0
        for i in range(2, int(limit ** 0.5) + 1):
            if is_p[i]:
                step = i
                start = i * i
                is_p[start:limit+1:step] = bytearray(len(is_p[start:limit+1:step]))
        return [i for i in range(limit + 1) if is_p[i]]

    best_k = 0
    best_p = 2

    # Check small primes first
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
        k = check_run(p)
        if k > best_k:
            best_k = k
            best_p = p
        if time.time() - start_time > deadline - 10:
            break

    # Search larger primes in batches
    # Strategy: sieve up to increasing limits, check each prime
    search_limits = [10**4, 10**5, 10**6, 10**7, 10**8]
    prev_limit = 100
    
    for limit in search_limits:
        if time.time() - start_time > deadline - 15:
            break
        
        try:
            primes = sieve(limit)
        except MemoryError:
            break
        
        for p in primes:
            if p <= prev_limit:
                continue
            if time.time() - start_time > deadline - 5:
                break
            
            # Quick check: if p^2 + 1 is prime, k=0 immediately
            n1 = p * p + 1
            if is_prime(n1) and n1 > p:
                continue
            
            k = check_run(p)
            if k > best_k:
                best_k = k
                best_p = p
        
        prev_limit = limit

    # If we have time, do a focused search on larger primes
    # Try some specific large primes that might have good properties
    remaining_time = deadline - (time.time() - start_time)
    if remaining_time > 20:
        # Try primes near 10^6, 10^7, etc.
        for target in [10**6, 5*10**6, 10**7, 5*10**7, 10**8]:
            if time.time() - start_time > deadline - 10:
                break
            # Find a prime near target
            start_p = target
            found = False
            for candidate in range(start_p, min(start_p + 10000, 10**9 + 1)):
                if is_prime(candidate):
                    n1 = candidate * candidate + 1
                    if not (is_prime(n1) and n1 > candidate):
                        k = check_run(candidate)
                        if k > best_k:
                            best_k = k
                            best_p = candidate
                    found = True
                    break
            if not found:
                # Try going down
                for candidate in range(start_p, max(start_p - 10000, 2), -1):
                    if is_prime(candidate):
                        n1 = candidate * candidate + 1
                        if not (is_prime(n1) and n1 > candidate):
                            k = check_run(candidate)
                            if k > best_k:
                                best_k = k
                                best_p = candidate
                        break

    return {'p': best_p, 'k': best_k}

if __name__ == "__main__":
    res = construct()
    print(res)
