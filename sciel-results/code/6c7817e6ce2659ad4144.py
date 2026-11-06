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
    # Primes up to 200 for quick residual checking
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
        """Fast prime check: small trial division then MR"""
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

    def is_p_smooth_small(n, p):
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
                if not is_p_smooth_small(n, p):
                    return False
                return True
        if n == 1:
            return True
        if is_prime(n):
            return n <= p
        d = pollard_brent(n)
        if d == n:
            return n <= p
        if not is_p_smooth_small(d, p):
            return False
        return is_p_smooth_small(n // d, p)

    def check_run_smooth_sieve(p, max_k=200):
        """
        Smoothness sieve approach: For each i in 1..max_k, compute m_i = (p^2+i) with
        all prime factors <= 200 removed. Then check if m_i is p-smooth.
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
                pass
            elif is_prime(m):
                if m > p:
                    break
            else:
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
                            if not is_p_smooth_small(m, p):
                                break
                        else:
                            if not is_p_smooth_small(a, p) or not is_p_smooth_small(b, p):
                                break
                else:
                    if not is_p_smooth_small(m, p):
                        break
            k = i
        return k

    # Density heuristic: estimate expected run length for a candidate p
    # For n ≈ p^2, the Dickman parameter u = ln(n)/ln(p) ≈ 2
    # ρ(2) ≈ 0.306 is the probability a random number near p^2 is p-smooth
    # But we need ALL of p^2+1..p^2+200 to be p-smooth
    # Better heuristic: count how many of p^2+1..p^2+200 are NOT prime
    # (since primes > p fail immediately) and estimate smoothness of composites
    def density_score(p, max_k=200):
        """
        Quick score: count how many of p^2+1..p^2+max_k are prime (bad)
        and estimate how many composites might have large prime factors.
        Lower score = more promising.
        """
        base = p * p
        prime_count = 0
        large_composite_count = 0
        for i in range(1, min(max_k + 1, 50)):  # Check first 50 for speed
            n = base + i
            # Quick primality check
            if is_prime(n):
                prime_count += 1
                continue
            # Remove small factors
            m = n
            for sp in primes_up_to_200:
                if sp * sp > m:
                    break
                while m % sp == 0:
                    m //= sp
            if m > 1:
                if is_prime(m):
                    if m > p:
                        large_composite_count += 1
                else:
                    # Composite with factors > 200; likely has a large factor
                    if m > p * p:  # Can't have all factors <= p if m > p^2
                        large_composite_count += 1
                    elif m > p:
                        # m is composite, all factors > 200, m in (200, p)
                        # Could still be p-smooth if it factors into primes <= p
                        # But for heuristic, count as potentially bad
                        large_composite_count += 0.5
        # Score: higher is worse
        return prime_count * 10 + large_composite_count

    # Generate candidate primes and test them
    best_k = 0
    best_p = 2

    # 1. Test small primes first (up to 10000)
    for p in small_primes:
        if p > 10000 and time.time() > deadline - 30:
            break
        k = check_run_smooth_sieve(p)
        if k > best_k:
            best_k = k
            best_p = p

    # 2. Search larger primes using segmented sieve with density-based prioritization
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
    # Use density heuristic to prioritize candidates
    current_low = 10**4
    current_high = 10**5

    while current_high <= 10**9 and time.time() < deadline - 15:
        candidates = segmented_sieve(current_low, current_high)

        # Score candidates using density heuristic (check first 50 values)
        # Only do full check on top-scoring candidates
        scored_candidates = []
        for p in candidates:
            if time.time() > deadline - 10:
                break
            score = density_score(p)
            scored_candidates.append((score, p))

        # Sort by score (lower is better) and check top candidates
        scored_candidates.sort(key=lambda x: x[0])

        # Check top 20% of candidates (or at least 50)
        num_to_check = max(50, len(scored_candidates) // 5)
        for score, p in scored_candidates[:num_to_check]:
            if time.time() > deadline - 5:
                break

            # Quick pre-check: test first 3 numbers using smooth sieve
            base = p * p
            valid = True
            for i in range(1, 4):
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
                        valid = False
                        break
                else:
                    if m < 10**12:
                        d = pollard_brent(m)
                        if d == m:
                            if m > p:
                                valid = False
                                break
                        else:
                            a, b = d, m // d
                            if (a > p and is_prime(a)) or (b > p and is_prime(b)):
                                valid = False
                                break
                            if not is_p_smooth_small(m, p):
                                valid = False
                                break
                    else:
                        if not is_p_smooth_small(m, p):
                            valid = False
                            break

            if not valid:
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
