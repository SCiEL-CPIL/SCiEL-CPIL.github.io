import time
import math
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    # Deterministic Miller-Rabin for n < 2^64
    MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        for p in MR_BASES:
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
        if n % 7 == 0:
            return 7
        if n % 11 == 0:
            return 11
        if n % 13 == 0:
            return 13
        if n % 17 == 0:
            return 17
        if n % 19 == 0:
            return 19
        if n % 23 == 0:
            return 23
        if n % 29 == 0:
            return 29
        if n % 31 == 0:
            return 31
        if n % 37 == 0:
            return 37
        while True:
            y = random.randrange(1, n)
            c = random.randrange(1, n)
            m = 128
            g, q, r = 1, 1, 1
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
                while True:
                    ys = (ys * ys + c) % n
                    g = math.gcd(abs(x - ys), n)
                    if g > 1:
                        break
            if g != n:
                return g

    def sieve_primes(limit):
        if limit < 2:
            return []
        sieve = bytearray(b'\x01') * (limit + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
        return [i for i, is_p in enumerate(sieve) if is_p]

    # Precompute primes up to 200000 for trial division
    TRIAL_LIMIT = 200000
    trial_primes = sieve_primes(TRIAL_LIMIT)

    def has_large_factor(n, limit):
        """
        Returns True if n has a prime factor > limit.
        Uses trial division first, then Miller-Rabin, then Pollard-Brent.
        """
        if n <= 1:
            return False
        if n <= limit:
            return False
        
        # Trial division by precomputed primes up to min(limit, TRIAL_LIMIT)
        upper = min(limit, TRIAL_LIMIT)
        for p in trial_primes:
            if p > upper:
                break
            if n % p == 0:
                while n % p == 0:
                    n //= p
                if n == 1:
                    return False
                if n <= limit:
                    return False
                # n still has factors; continue checking
                break
        
        if n == 1 or n <= limit:
            return False
        
        # Check if n is prime
        if is_prime(n):
            return n > limit
        
        # n is composite and > limit
        # Try to factor it
        d = pollard_brent(n)
        if d == n:
            # Should not happen if n is composite, but safety
            return n > limit
        return has_large_factor(d, limit) or has_large_factor(n // d, limit)

    def check_run(p, max_k=200):
        """
        Returns k such that p^2+1 ... p^2+k are all p-smooth.
        """
        k = 0
        base = p * p
        while k < max_k:
            n = base + k + 1
            if has_large_factor(n, p):
                break
            k += 1
        return k

    def segmented_sieve(low, high):
        if high < 2:
            return []
        low = max(low, 2)
        if low % 2 == 0:
            low += 1
        if high % 2 == 0:
            high -= 1
        
        limit = int(math.isqrt(high)) + 1
        small_primes = sieve_primes(limit)
        
        size = high - low + 1
        is_prime_seg = bytearray(b'\x01') * size
        if low == 1:
            is_prime_seg[0] = 0
        if low == 2:
            is_prime_seg[0] = 1
        else:
            is_prime_seg[0] = 0
        
        if low % 2 != 0:
            for i in range(1, size, 2):
                is_prime_seg[i] = 0
        
        for p in small_primes:
            if p == 2:
                continue
            start = ((low + p - 1) // p) * p
            if start == p:
                start += p
            for j in range(start - low, size, p):
                is_prime_seg[j] = 0
        
        result = []
        for i in range(size):
            if is_prime_seg[i]:
                result.append(low + i)
        return result

    best_k = 0
    best_p = 2

    # Phase 1: Check small primes quickly to establish baseline
    primes_small = sieve_primes(10000)
    for p in primes_small:
        if time.time() > deadline:
            break
        k = check_run(p)
        if k > best_k:
            best_k = k
            best_p = p

    # Phase 2: Search medium primes [10000, 10^6]
    current_low = 10001
    chunk_size = 50000
    while current_low < 10**6:
        if time.time() > deadline:
            break
        current_high = min(current_low + chunk_size - 1, 10**6)
        primes_chunk = segmented_sieve(current_low, current_high)
        
        for p in primes_chunk:
            if time.time() > deadline:
                break
            k = check_run(p)
            if k > best_k:
                best_k = k
                best_p = p
        
        current_low = current_high + 1

    # Phase 3: Search larger primes [10^6, 10^9] with random sampling
    # Larger primes are more likely to have long runs
    if time.time() < deadline - 10:
        # Try random primes in increasing ranges
        ranges = [
            (10**6, 10**7),
            (10**7, 10**8),
            (10**8, 10**9),
        ]
        
        for low, high in ranges:
            if time.time() > deadline:
                break
            # Try a batch of random primes in this range
            attempts = 0
            max_attempts = 500
            while attempts < max_attempts and time.time() < deadline:
                x = random.randrange(low, high)
                if x % 2 == 0:
                    x += 1
                # Find next prime
                while not is_prime(x):
                    x += 2
                    if x > high:
                        break
                if x > high:
                    break
                attempts += 1
                k = check_run(x)
                if k > best_k:
                    best_k = k
                    best_p = x

    return {'p': best_p, 'k': best_k}
