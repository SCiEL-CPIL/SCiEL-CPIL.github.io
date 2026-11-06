import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    def is_prime(n):
        if n < 2:
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

    small_primes = sieve_primes(10**4)

    def is_p_smooth(n, p):
        if n <= 1:
            return True
        if is_prime(n):
            return n <= p
        for r in small_primes:
            if r > p:
                break
            if n % r == 0:
                while n % r == 0:
                    n //= r
                if n == 1:
                    return True
                if is_prime(n):
                    return n <= p
        if n == 1:
            return True
        if is_prime(n):
            return n <= p
        d = pollard_brent(n)
        return is_p_smooth(d, p) and is_p_smooth(n // d, p)

    def check_run(p, max_k=200):
        k = 0
        base = p * p
        while k < max_k:
            n = base + k + 1
            if not is_p_smooth(n, p):
                break
            k += 1
        return k

    best_k = 0
    best_p = 2

    # Phase 1: primes up to 10^6
    primes_1m = sieve_primes(10**6)
    for p in primes_1m:
        if time.time() > deadline:
            break
        k = check_run(p)
        if k > best_k:
            best_k = k
            best_p = p

    # Phase 2: segmented sieve for [10^6, 10^8]
    def segmented_sieve(low, high):
        if high < 2:
            return []
        low = max(low, 2)
        if low % 2 == 0:
            low += 1
        if high % 2 == 0:
            high -= 1
        limit = int(math.isqrt(high)) + 1
        base_primes = sieve_primes(limit)
        size = high - low + 1
        is_prime_seg = bytearray(b'\x01') * size
        for p in base_primes:
            if p * p > high:
                break
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

    if time.time() < deadline - 20:
        chunk_size = 10**6
        current_low = 10**6 + 1
        while current_low < 10**8:
            if time.time() > deadline:
                break
            current_high = min(current_low + chunk_size - 1, 10**8)
            primes_chunk = segmented_sieve(current_low, current_high)
            for p in primes_chunk:
                if time.time() > deadline:
                    break
                k = check_run(p)
                if k > best_k:
                    best_k = k
                    best_p = p
            current_low = current_high + 1

    # Phase 3: random large primes
    if time.time() < deadline - 10:
        for _ in range(200):
            if time.time() > deadline:
                break
            x = random.randrange(10**8, 10**9)
            if x % 2 == 0:
                x += 1
            while not is_prime(x):
                x += 2
                if x > 10**9:
                    break
            if x > 10**9:
                continue
            k = check_run(x)
            if k > best_k:
                best_k = k
                best_p = x

    return {'p': best_p, 'k': best_k}
