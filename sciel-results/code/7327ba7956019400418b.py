import time
import random
import math
from bisect import bisect_right

def construct():
    start_time = time.time()
    DEADLINE = start_time + 115.0  # Safety margin before 150s kill

    # Precompute primes up to 10^6 using sieve
    SIEVE_LIMIT = 1000000
    is_prime_sieve = bytearray(b'\x01') * (SIEVE_LIMIT + 1)
    is_prime_sieve[0] = 0
    is_prime_sieve[1] = 0
    for i in range(2, int(math.isqrt(SIEVE_LIMIT)) + 1):
        if is_prime_sieve[i]:
            for j in range(i * i, SIEVE_LIMIT + 1, i):
                is_prime_sieve[j] = 0
    primes = [i for i in range(2, SIEVE_LIMIT + 1) if is_prime_sieve[i]]
    
    # For Miller-Rabin, bases sufficient for n < 3,317,444,400,000,000,000
    MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    def is_prime_mr(n):
        if n < 2:
            return False
        if n <= SIEVE_LIMIT:
            return bool(is_prime_sieve[n])
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n == p:
                return True
            if n % p == 0:
                return False
        # Write n-1 as d * 2^r
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        for a in MR_BASES:
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

    def pollard_rho(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        if n % 5 == 0:
            return 5
        while True:
            x = random.randint(2, n - 1)
            y = x
            c = random.randint(1, n - 1)
            d = 1
            while d == 1:
                x = (x * x + c) % n
                y = (y * y + c) % n
                y = (y * y + c) % n
                d = math.gcd(abs(x - y), n)
            if d != n:
                return d

    def factorize(n):
        """Factor n using trial division by small primes, then Pollard's Rho."""
        if n == 1:
            return []
        if n <= SIEVE_LIMIT:
            if is_prime_sieve[n]:
                return [n]
            # Find prime factors by trial division
            result = []
            temp = n
            for p in primes:
                if p * p > temp:
                    break
                while temp % p == 0:
                    result.append(p)
                    temp //= p
            if temp > 1:
                result.append(temp)
            return result
        
        # First, try trial division by primes up to 10^6
        result = []
        temp = n
        for p in primes:
            if p * p > temp:
                break
            while temp % p == 0:
                result.append(p)
                temp //= p
        
        if temp > 1:
            if temp <= SIEVE_LIMIT:
                if is_prime_sieve[temp]:
                    result.append(temp)
                else:
                    # Shouldn't happen, but handle it
                    for p in primes:
                        if p * p > temp:
                            break
                        while temp % p == 0:
                            result.append(p)
                            temp //= p
                    if temp > 1:
                        result.append(temp)
            else:
                # Use Pollard's Rho for the remaining composite
                if is_prime_mr(temp):
                    result.append(temp)
                else:
                    result.extend(factorize(temp))
        return result

    def phi(n):
        if n == 1:
            return 1
        factors = factorize(n)
        unique_factors = set(factors)
        result = n
        for p in unique_factors:
            result -= result // p
        return result

    best_k = 1
    best_s = 1

    def check_start(s):
        nonlocal best_k, best_s
        seen = set()
        k = 0
        for i in range(200):
            val = phi(s + i)
            if val in seen:
                break
            seen.add(val)
            k += 1
        if k > best_k:
            best_k = k
            best_s = s
        return k

    # Phase 1: Deterministic scan of small numbers
    # Small numbers are fast to factor. Scan up to 200,000.
    scan_limit = 200000
    for s in range(1, scan_limit + 1):
        if time.time() > DEADLINE:
            break
        check_start(s)
        if best_k == 200:
            return {'start': best_s, 'length': best_k}

    # Phase 2: Random sampling in larger ranges
    random.seed(42)
    
    # Sample starts in [10^6, 10^12]
    while time.time() < DEADLINE - 5.0:
        s = random.randint(1000000, 10**12 - 200)
        k_found = check_start(s)
        
        if best_k == 200:
            break
            
        # If we found a moderately long run, explore neighbors
        if k_found > 30:
            for delta in range(-15, 16):
                if delta == 0:
                    continue
                new_s = best_s + delta
                if 1 <= new_s and new_s + 199 <= 10**12:
                    if time.time() > DEADLINE - 2.0:
                        break
                    check_start(new_s)
                    if best_k == 200:
                        break
            if best_k == 200:
                break

    return {'start': best_s, 'length': best_k}
