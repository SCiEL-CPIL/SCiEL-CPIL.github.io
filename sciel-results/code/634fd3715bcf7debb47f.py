import math
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140  # 10s margin before 150s kill
    
    # Deterministic Miller-Rabin for 64-bit
    MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    def is_prime(n):
        if n < 2:
            return False
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n % p == 0:
                return n == p
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for a in MR_BASES:
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
            g = 1
            r = 1
            q = 1
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
                    if g != 1 and g != n:
                        return g
            else:
                return g

    def factorize(n):
        factors = []
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
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
            x = stack.pop()
            if is_prime(x):
                factors.append(x)
            else:
                d = pollard_brent(x)
                if d == x:
                    factors.append(x)
                else:
                    stack.append(d)
                    stack.append(x // d)
        return factors

    def all_factors_leq(n, bound):
        if n <= 1:
            return True
        # Quick primality check
        if is_prime(n):
            return n <= bound
        # Factorize and check
        for f in factorize(n):
            if f > bound:
                return False
        return True

    # Generate primes up to 500k for initial search
    # Then try random large primes
    limit_small = 500000
    sieve = bytearray(b'\x01') * (limit_small + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit_small ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    
    small_primes = [i for i, v in enumerate(sieve) if v]
    
    best_k = 0
    best_p = 7  # Fallback
    found_small = False
    
    for p in small_primes:
        if time.time() - start_time > DEADLINE:
            break
        k = 0
        for offset in range(1, 201):
            n = p * p + offset
            if all_factors_leq(n, p):
                k += 1
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p
            if k >= 5:
                found_small = True
        # Skip if we've been checking for a long time
        if time.time() - start_time > DEADLINE - 20:
            break
    
    # Now try random primes in larger ranges
    # Focus on ranges where p^2 + i might have small factors
    # We'll try random primes up to 10^9
    attempts = 0
    max_attempts = 20000
    
    while time.time() - start_time < DEADLINE - 5 and attempts < max_attempts:
        attempts += 1
        # Generate a random candidate prime
        # Bias towards larger p but not too large
        # p^2 can be up to 10^18, which is 60 bits
        # Random 30-45 bit numbers
        target_bits = random.randint(30, 45)
        x = random.getrandbits(target_bits) | 1  # Make odd
        # Find next prime
        while not is_prime(x):
            x += 2
            if time.time() - start_time > DEADLINE - 5:
                break
        if time.time() - start_time > DEADLINE - 5:
            break
        if x > 10**9:
            continue
        if x < 2:
            continue
        
        # Check this p
        k = 0
        for offset in range(1, 201):
            n = x * x + offset
            if all_factors_leq(n, x):
                k += 1
            else:
                break
        if k > best_k:
            best_k = k
            best_p = x
    
    # Ensure valid result
    if best_k < 1:
        best_p = 7
        best_k = 1
    
    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
