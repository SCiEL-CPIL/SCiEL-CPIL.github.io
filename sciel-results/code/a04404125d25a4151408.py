import math
import time

def construct():
    start_time = time.time()
    DEADLINE = start_time + 145.0  # safety margin before 150s limit

    def sieve_primes(limit):
        if limit < 2:
            return []
        is_p = bytearray(b'\x01') * (limit + 1)
        is_p[0] = is_p[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if is_p[i]:
                is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
        return [i for i in range(2, limit + 1) if is_p[i]]

    # Sieve primes up to 3e6: used for trial division, segmented sieve, and candidates
    SIEVE_LIMIT = 3000000
    primes = sieve_primes(SIEVE_LIMIT)
    # primes up to 10^5 for fast modular pre-filter (these divide p^2+i for some i<=200)
    small_primes = [q for q in primes if q <= 100000]

    # Deterministic Miller-Rabin for n < 2^64
    def is_prime_mr(n):
        if n < 2:
            return False
        if n in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1
            d //= 2
        for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
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

    # Pollard-Brent
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        if n % 5 == 0:
            return 5
        while True:
            y, c, m = 2, 1, 128
            g, r, q = 1, 0, 1
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
                r += 1
            if g == n:
                g = 1
                while g == 1:
                    ys = (ys * ys + c) % n
                    g = math.gcd(abs(x - ys), n)
            if g != n:
                return g
            c += 1

    def is_p_smooth(n, p, primes):
        """Check if all prime factors of n are <= p."""
        if n <= 1:
            return True
        m = n
        for q in primes:
            if q > p:
                break
            if q * q > m:
                break
            while m % q == 0:
                m //= q
        if m == 1:
            return True
        if m <= p:
            return True
        # m has a factor > p
        if is_prime_mr(m):
            return False
        d = pollard_brent(m)
        return is_p_smooth(d, p, primes) and is_p_smooth(m // d, p, primes)

    def check_run_segmented(p, primes, deadline):
        """Check run length from p^2+1 using segmented sieve + Pollard-Brent for remainders."""
        if time.time() > deadline:
            return 0
        p_sq = p * p
        W = 200
        # Segmented sieve over [p^2+1, p^2+W]
        rem = [p_sq + i for i in range(1, W + 1)]
        # Sieve out all small primes q <= p
        for q in primes:
            if q > p:
                break
            if q * q > rem[-1]:
                break
            # find first index j in [1..W] with (p_sq + j) % q == 0
            r0 = p_sq % q
            first = (q - r0) % q
            if first == 0:
                first = q
            if first > W:
                continue
            for j in range(first, W + 1, q):
                while rem[j] % q == 0:
                    rem[j] //= q
        # Now rem[i-1] is the part of p^2+i not divisible by any prime <= p (or <= sqrt)
        k = 0
        for i in range(1, W + 1):
            if time.time() > deadline:
                break
            m = rem[i - 1]
            if m == 1:
                k += 1
                continue
            if m <= p:
                k += 1
                continue
            # m has a factor > p; verify
            if is_prime_mr(m):
                break
            d = pollard_brent(m)
            if is_p_smooth(d, p, primes) and is_p_smooth(m // d, p, primes):
                k += 1
            else:
                break
        return k

    def modular_prefilter(p):
        """Cheap test: for each small prime q (q<=1e5), check that no i in 1..200
        has q | (p^2+i) AND q > p (impossible since q<=1e5<p for large p) -- instead
        we check: for q <= p, q | (p^2+i) is fine (q is a small factor, OK).
        The real filter: for q > p (q in small_primes with q > p), if q | (p^2+i) for some i,
        then p^2+i has a prime factor q > p -> fail. So check no such i exists."""
        r = (p * p) % 1  # placeholder
        for q in small_primes:
            if q <= p:
                continue
            r = (p * p) % q
            # need i in [1,200] with (r + i) % q == 0
            i = (q - r) % q
            if i == 0:
                i = q
            if i <= 200:
                return False
        return True

    best_k = 0
    best_p = 2

    # Phase 1: primes <= SIEVE_LIMIT (full segmented check)
    for p in primes:
        if time.time() > DEADLINE - 5:
            break
        if p < 3:
            continue
        k = check_run_segmented(p, primes, DEADLINE - 2)
        if k > best_k:
            best_k = k
            best_p = p

    # Phase 2: larger primes. Use modular pre-filter then segmented check.
    # For p > SIEVE_LIMIT, primes list (up to 3e6) still covers trial division up to 3e6,
    # which is enough to remove all small factors from p^2+i (since p^2+i ~ p^2, sqrt ~ p;
    # factors between 3e6 and p are possible but Pollard-Brent handles the remainder).
    if time.time() < DEADLINE - 10:
        # Search candidates in a range above SIEVE_LIMIT using step 2, MR-filtered
        lo = SIEVE_LIMIT + 1
        if lo % 2 == 0:
            lo += 1
        hi = 10**9
        # We can't iterate all; sample a window and also try structured candidates.
        # Try a window of candidates near SIEVE_LIMIT first (dense primes).
        window_end = min(lo + 2000000, hi)
        for candidate in range(lo, window_end, 2):
            if time.time() > DEADLINE - 5:
                break
            if not is_prime_mr(candidate):
                continue
            if not modular_prefilter(candidate):
                continue
            k = check_run_segmented(candidate, primes, DEADLINE - 2)
            if k > best_k:
                best_k = k
                best_p = candidate

    return {'p': best_p, 'k': best_k}
