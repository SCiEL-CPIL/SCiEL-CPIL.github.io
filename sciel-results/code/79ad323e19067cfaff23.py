"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random
import sys

def construct():
    deadline = time.time() + 110.0  # safety margin before 150s kill

    # Precompute small primes up to 10^6 for trial division
    def sieve_primes(n):
        if n < 2:
            return []
        bs = bytearray(b'\x01') * (n + 1)
        bs[0] = bs[1] = 0
        for i in range(2, int(math.isqrt(n)) + 1):
            if bs[i]:
                bs[i*i::i] = bytearray(len(bs[i*i::i]))
        return [i for i in range(2, n + 1) if bs[i]]

    small_primes = sieve_primes(10**6)
    primes_up_to_200 = [p for p in small_primes if p <= 200]

    # Deterministic Miller-Rabin for n < 3.2e18
    def is_prime_mr(n):
        if n < 2:
            return False
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n % p == 0:
                return n == p
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
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

    def is_prime(n):
        """Fast prime check"""
        if n < 2:
            return False
        if n in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            return True
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n % p == 0:
                return False
        if n < 10**6:
            limit = int(math.isqrt(n))
            for sp in small_primes:
                if sp > limit:
                    break
                if n % sp == 0:
                    return False
            return True
        return is_prime_mr(n)

    # Pollard's rho with Brent's improvement
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        if n % 5 == 0:
            return 5
        while True:
            c = random.randrange(1, n - 1)
            y = random.randrange(0, n - 1)
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

    def is_p_smooth(n, p):
        """Check if all prime factors of n are <= p"""
        if n <= 1:
            return True
        if is_prime(n):
            return n <= p
        for sp in small_primes:
            if sp * sp > n:
                break
            if sp > p:
                break
            if n % sp == 0:
                n //= sp
                if not is_p_smooth(n, p):
                    return False
                return True
        if n == 1:
            return True
        if is_prime(n):
            return n <= p
        d = pollard_brent(n)
        if d == n:
            return n <= p
        if not is_p_smooth(d, p):
            return False
        return is_p_smooth(n // d, p)

    def check_run_smooth_sieve(p, max_k=200):
        """
        For each i in 1..max_k, compute m_i = (p^2+i) with all prime factors <= 200 removed.
        Then check if m_i is p-smooth.
        """
        base = p * p
        k = 0
        for i in range(1, max_k + 1):
            n = base + i
            m = n
            # Remove all prime factors <= 200
            for sp in primes_up_to_200:
                if sp * sp > m:
                    break
                while m % sp == 0:
                    m //= sp
            # Now m has no prime factors <= 200
            if m == 1:
                pass  # p-smooth
            elif is_prime(m):
                if m > p:
                    break  # not p-smooth
            else:
                # m is composite with all prime factors > 200
                if m < 10**14:
                    d = pollard_brent(m)
                    if d == m:
                        if m > p:
                            break
                    else:
                        a, b = d, m // d
                        if a > p or b > p:
                            large = max(a, b)
                            if is_prime(large) and large > p:
                                break
                            if not is_p_smooth(m, p):
                                break
                        else:
                            if not is_p_smooth(a, p) or not is_p_smooth(b, p):
                                break
                else:
                    if not is_p_smooth(m, p):
                        break
            k = i
        return k

    # Quick pre-check: test first few numbers to quickly reject most candidates
    def quick_precheck(p, base, num_checks=3):
        """Quickly check if first num_checks numbers are p-smooth"""
        for i in range(1, num_checks + 1):
            n = base + i
            m = n
            for sp in primes_up_to_200:
                if sp * sp > m:
                    break
                while m % sp == 0:
                    m //= sp
            if m == 1:
                continue
            if is_prime(m):
                if m > p:
                    return False
            else:
                # Composite with factors > 200, need deeper check
                if m < 10**12:
                    d = pollard_brent(m)
                    if d == m:
                        if m > p:
                            return False
                    else:
                        a, b = d, m // d
                        if (a > p and is_prime(a)) or (b > p and is_prime(b)):
                            return False
                        if not is_p_smooth(m, p):
                            return False
                else:
                    if not is_p_smooth(m, p):
                        return False
        return True

    # Generate candidate primes and test them
    best_k = 0
    best_p = 2

    # 1. Test small primes first
    for p in small_primes:
        if p > 10000 and time.time() > deadline - 30:
            break
        k = check_run_smooth_sieve(p)
        if k > best_k:
            best_k = k
            best_p = p

    # 2. Search larger primes using segmented sieve
    def segmented_sieve(low, high):
        if high < 2:
            return []
        limit = int(math.isqrt(high)) + 1
        if limit <= 10**6:
            small = [p for p in small_primes if p <= limit]
        else:
            small = sieve_primes(limit)
        sieve = bytearray(b'\x01') * (high - low + 1)
        for p in small:
            if p * p > high:
                break
            start = max(p * p, ((low + p - 1) // p) * p)
            for j in range(start, high + 1, p):
                sieve[j - low] = 0
        primes = []
        for i in range(low, high + 1):
            if sieve[i - low] and i >= 2:
                primes.append(i)
        return primes

    # Search ranges - focus on moderate primes where runs are likely
    current_low = 10**4
    current_high = 10**5

    while current_high <= 10**9 and time.time() < deadline - 15:
        candidates = segmented_sieve(current_low, current_high)

        for p in candidates:
            if time.time() > deadline - 5:
                break

            # Quick pre-check: test first 3 numbers using smooth sieve
            base = p * p
            if not quick_precheck(p, base, 3):
                continue

            # If pre-check passes, do full check
            k = check_run_smooth_sieve(p)
            if k > best_k:
                best_k = k
                best_p = p

        # Increase range
        current_low = current_high
        if current_high < 10**6:
            current_high = current_high * 10
        else:
            current_high = current_high + 10**6

    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
