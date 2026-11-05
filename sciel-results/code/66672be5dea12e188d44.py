import math
import time
import random

def construct():
    start_time = time.time()
    LIMIT = 115.0  # Leave margin for overhead and evaluation setup
    
    # 1. Generate primes up to 10^6 using a fast bytearray sieve
    limit_p = 10**6
    is_prime = bytearray(b'\x01') * (limit_p + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(limit_p**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i, val in enumerate(is_prime) if val]
    
    # Miller-Rabin for primality testing of large numbers
    def is_prime_large(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True
    
    # 2. Efficient phi calculation
    def phi_fast(x):
        if x == 1:
            return 1
        result = x
        n = x
        # Divide by primes up to 10^4
        for p in primes:
            if p > 10000:
                break
            if n % p == 0:
                while n % p == 0:
                    n //= p
                result -= result // p
        if n > 1:
            if is_prime_large(n):
                result -= result // n
            else:
                # n is composite, must be product of two primes > 10^4
                limit = int(math.isqrt(n))
                found = False
                for p in primes:
                    if p <= 10000:
                        continue
                    if p > limit:
                        break
                    if n % p == 0:
                        while n % p == 0:
                            n //= p
                        result -= result // p
                        found = True
                        break
                if found and n > 1:
                    result -= result // n
                elif not found:
                    result -= result // n
        return result

    # 3. Search for a run of length 200 with distinct totients
    best_k = 1
    best_s = 1
    
    # Helper to check a run starting at s
    def check_run(s, target_len=200):
        seen = set()
        k = 0
        for i in range(target_len):
            if time.time() - start_time > LIMIT:
                break
            val = phi_fast(s + i)
            if val in seen:
                break
            seen.add(val)
            k += 1
        return k

    # First, try some deterministic starting points that are likely to work
    test_starts = [
        10**12 - 200,
        10**12 - 1000,
        10**12 - 5000,
        5 * 10**11,
        3 * 10**11,
        7 * 10**10,
        10**11,
        10**10,
    ]
    
    for s in test_starts:
        if s < 1 or s + 199 > 10**12:
            continue
        k = check_run(s, 200)
        if k > best_k:
            best_k = k
            best_s = s
            if best_k == 200:
                return {'start': best_s, 'length': best_k}
        
        if time.time() - start_time > LIMIT:
            break

    # If deterministic didn't find 200, try random sampling
    random.seed(42)
    attempts = 0
    max_attempts = 5000
    
    while time.time() - start_time < LIMIT and attempts < max_attempts:
        r = random.random()
        if r < 0.4:
            s = random.randint(10**12 - 10**6, 10**12 - 199)
        elif r < 0.7:
            s = random.randint(10**10, 10**12 - 199)
        else:
            s = random.randint(10**6, 10**10)
        
        k = check_run(s, 200)
        
        if k > best_k:
            best_k = k
            best_s = s
            if best_k == 200:
                break
        
        attempts += 1

    return {'start': best_s, 'length': best_k}
