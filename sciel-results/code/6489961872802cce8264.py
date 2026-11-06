"""Rewritten program using segmented sieve for SPF over [p^2+1, p^2+200]."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    # Find a prime p <= 1e9 and k (1..200) such that
    # every integer in [p^2+1, p^2+k] is p-smooth.
    # Maximize k.

    # Core idea:
    # For each candidate prime p, we only need to inspect 200 consecutive integers.
    # We can batch-check p-smoothness by segmented trial division over those 200 values.
    #
    # Efficient segmented check:
    #   Let A[i] = p^2 + 1 + i for i in 0..199.
    #   For each prime q <= p, if any A[i] is divisible by q, divide out all powers of q.
    #   At the end, A[i] == 1 iff p^2+1+i is p-smooth.
    #
    # We can stop early when the current run length can no longer beat the best.
    #
    # To cover large p efficiently:
    #   - Precompute primes up to B (base bound) for fast small-prime division.
    #   - For p <= B, use all primes <= p directly.
    #   - For p > B, use primes <= B plus a Miller-Rabin / Pollard-Brent fallback
    #     only on the remaining cofactors (which are rare and small-count).

    def sieve_primes(limit):
        """Return list of primes up to limit using a simple bytearray sieve."""
        if limit < 2:
            return []
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        r = int(math.isqrt(limit))
        for i in range(2, r + 1):
            if is_prime[i]:
                start = i * i
                step = i
                is_prime[start::step] = bytearray(len(is_prime[start::step]))
        return [i for i in range(2, limit + 1) if is_prime[i]]

    def is_probable_prime(n):
        """Deterministic Miller-Rabin for n < 3,317,044,064,679,887,385,961,981."""
        if n < 2:
            return False
        small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
        for p in small_primes:
            if n % p == 0:
                return n == p
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        # Bases sufficient for 64-bit integers
        for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
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
        """Return a non-trivial factor of composite n (n > 1)."""
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = 2
            c = 1
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
                g = 1
                while g == 1:
                    ys = (ys * ys + c) % n
                    g = math.gcd(abs(x - ys), n)
            if g != n:
                return g

    def factorize(n, small_primes, small_limit):
        """
        Factor n using small_primes up to small_limit, then Miller-Rabin/Pollard-Brent.
        Returns list of prime factors with multiplicity.
        """
        factors = []
        for q in small_primes:
            if q > small_limit:
                break
            if q * q > n:
                break
            while n % q == 0:
                factors.append(q)
                n //= q
        if n == 1:
            return factors
        if is_probable_prime(n):
            factors.append(n)
            return factors
        stack = [n]
        while stack:
            m = stack.pop()
            if m == 1:
                continue
            if is_probable_prime(m):
                factors.append(m)
                continue
            d = pollard_brent(m)
            stack.append(d)
            stack.append(m // d)
        return factors

    # Choose base prime bound.
    # We want to trial-divide by many small primes quickly.
    # 200k primes is very fast to sieve and use.
    B = 200_000
    base_primes = sieve_primes(B)

    best_k = 0
    best_p = 0

    start_time = time.time()
    deadline = start_time + 115.0  # leave 5s margin

    # Phase 1: exhaustive over primes up to B using segmented trial division.
    for p in base_primes:
        if time.time() > deadline:
            break
        p2 = p * p

        # A[i] = p^2 + 1 + i
        A = [p2 + 1 + i for i in range(200)]

        # Divide out primes q <= p.
        # Since p <= B, base_primes contains all needed q.
        # We can stop dividing when q > p.
        for q in base_primes:
            if q > p:
                break
            # For each index where A[i] % q == 0, divide out q.
            # We can vectorize-ish with a loop over indices; 200 is tiny.
            for i in range(200):
                if A[i] % q == 0:
                    while A[i] % q == 0:
                        A[i] //= q

        # Now A[i] == 1 iff p-smooth.
        k = 0
        for i in range(200):
            if A[i] == 1:
                k += 1
            else:
                break

        if k > best_k:
            best_k = k
            best_p = p

    # Phase 2: larger primes p > B, up to some cap, with hybrid segmented + fallback.
    # For p > B, we cannot precompute all primes <= p.
    # But we can still divide by base_primes (<= B).
    # After that, remaining cofactors are either 1 or have all prime factors > B.
    # Since n ≈ p^2, any remaining cofactor > 1 is likely a prime > B or product of two primes > B.
    # We can factor the remaining cofactor with Miller-Rabin/Pollard-Brent and check max factor <= p.
    #
    # To keep runtime bounded, we sample primes in larger ranges.
    # We generate primes in chunks using a segmented sieve over [lo, hi].

    def segmented_primes(lo, hi, base_primes):
        """Generate primes in [lo, hi] using base_primes."""
        if hi < 2:
            return []
        lo = max(lo, 2)
        size = hi - lo + 1
        is_prime = bytearray(b'\x01') * size
        for q in base_primes:
            if q * q > hi:
                break
            start = ((lo + q - 1) // q) * q
            if start == q:
                start += q
            for j in range(start, hi + 1, q):
                is_prime[j - lo] = 0
        return [lo + i for i in range(size) if is_prime[i]]

    # Try larger primes in increasing chunks until deadline.
    chunk_size = 100_000
    lo = B + 1
    while lo <= 10_000_000 and time.time() < deadline:
        hi = min(lo + chunk_size - 1, 10_000_000)
        primes_chunk = segmented_primes(lo, hi, base_primes)
        for p in primes_chunk:
            if time.time() > deadline:
                break
            p2 = p * p
            A = [p2 + 1 + i for i in range(200)]

            # Divide by base primes (all <= B < p)
            for q in base_primes:
                # q <= B < p, so always allowed
                for i in range(200):
                    if A[i] % q == 0:
                        while A[i] % q == 0:
                            A[i] //= q

            # Now check p-smoothness: remaining A[i] must have all prime factors <= p.
            k = 0
            for i in range(200):
                rem = A[i]
                if rem == 1:
                    k += 1
                    continue
                # Factor rem (small count, usually prime or product of two primes)
                facs = factorize(rem, base_primes, B)
                if any(f > p for f in facs):
                    break
                k += 1

            if k > best_k:
                best_k = k
                best_p = p

        lo = hi + 1

    if best_k >= 1:
        return {'p': best_p, 'k': best_k}

    return {'p': 3, 'k': 1}

# EVOLVE-BLOCK-END
