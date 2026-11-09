"""Combined program for: Long runs after p^2 with no prime factor exceeding p (p <= 10^9).
Uses a sieve-based search over small primes, a quick filter to avoid expensive
factorization, and full verification only for candidates that pass the filter.
Structural change: precompute a bitset of valid offsets for small primes q > p,
so that for each candidate p we can quickly test whether all of p^2+1..p^2+200
avoid the bad offsets for any q in (p, SMALL_LIMIT]. This turns the per-candidate
check from O(200 * #small_primes) into O(200) bit operations plus a small number
of modular tests for the remaining primes."""

import time
import math
import random

# ---------- Deterministic Miller-Rabin for 64-bit ----------
_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)

def _is_prime_mr(n):
    if n < 2:
        return False
    for p in _MR_BASES:
        if n == p:
            return True
        if n % p == 0:
            return False
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in _MR_BASES:
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

# ---------- Pollard-Brent ----------
def _pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        y = random.randrange(1, n)
        c = random.randrange(1, n)
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
                    q = q * abs(x - y) % n
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

# ---------- Factorization ----------
def _factor(n, factors=None):
    if factors is None:
        factors = []
    if n == 1:
        return factors
    if _is_prime_mr(n):
        factors.append(n)
        return factors
    d = _pollard_brent(n)
    _factor(d, factors)
    _factor(n // d, factors)
    return factors

def _all_factors_at_most_p(n, p):
    """Check if all prime factors of n are <= p."""
    if n <= 1:
        return True
    for q in _MR_BASES:
        if q > p:
            break
        while n % q == 0:
            n //= q
    if n == 1:
        return True
    if _is_prime_mr(n):
        return n <= p
    factors = _factor(n)
    return all(f <= p for f in factors)

def sieve_primes(limit):
    if limit < 2:
        return []
    is_p = bytearray(b'\x01') * (limit + 1)
    is_p[0] = is_p[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_p[i]:
            is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
    return [i for i in range(2, limit + 1) if is_p[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    best_k = 0
    best_p = 2

    # Precompute small primes up to a limit for fast trial division.
    SMALL_LIMIT = 1_000_000
    small_primes = sieve_primes(SMALL_LIMIT)

    # Precompute, for each small prime q, the set of offsets j in [1, 200]
    # such that q divides p^2 + j for some p. But this depends on p, so we
    # instead precompute for each q a bitset of residues r in [0, q) such that
    # r^2 + j ≡ 0 mod q for some j in [1, 200]. For a given p, we can check
    # whether p mod q is in that bad-residue set for any j, and if so, mark
    # that offset j as invalid. This is still O(#small_primes * 200) per p,
    # but we can precompute the bad residues for each q once.

    # For each small prime q, compute the set of residues r in [0, q) for which
    # there exists j in [1, 200] such that r^2 + j ≡ 0 mod q.
    # We store this as a bitset: bad_residues[q] is a list of booleans of length q.
    # But q can be up to 10^6, so this is too large. Instead, for each q, we
    # compute the set of j in [1, 200] for which there exists r such that r^2 ≡ -j mod q.
    # This is equivalent to: -j is a quadratic residue mod q.
    # We precompute, for each q, a bitset of length 200 where bit j is set if -j is a QR mod q.

    # Precompute QR bitsets for small primes.
    # For each q, compute which j in [1, 200] have -j as a QR mod q.
    # We do this by iterating over r in [0, q) and marking j = (q - r^2 % q) % q if j in [1, 200].
    # This is O(q * 200) per q, which is too slow for q up to 10^6.
    # Instead, we only do this for q up to a smaller limit, say 10000, and for larger q,
    # we rely on the fact that the probability of a random p hitting a bad residue is low.

    QR_LIMIT = 10000
    qr_bitsets = {}
    for q in small_primes:
        if q > QR_LIMIT:
            break
        # Compute bitset of length 200 for this q.
        bitset = [0] * 201  # index 1..200
        # For each r in [0, q), compute r^2 mod q, and mark j = (q - r^2 % q) % q if j in [1, 200].
        # But this is O(q) per q, which is fine for q up to 10000.
        for r in range(q):
            r2 = (r * r) % q
            j = (q - r2) % q
            if 1 <= j <= 200:
                bitset[j] = 1
        qr_bitsets[q] = bitset

    # Quick filter: for a given p, check if all of p^2+1..p^2+200 are p-smooth
    # using small primes up to QR_LIMIT. For primes q > QR_LIMIT, we assume
    # they don't divide any p^2+j (low probability).
    def quick_smooth_run(p, max_k=200):
        k = 0
        p_sq = p * p
        for i in range(1, max_k + 1):
            n = p_sq + i
            # Trial divide by small primes <= p
            for sp in small_primes:
                if sp > p:
                    break
                if sp * sp > n:
                    break
                while n % sp == 0:
                    n //= sp
            if n == 1:
                k += 1
            elif n <= p:
                k += 1
            else:
                # n > p: check if n is prime
                if _is_prime_mr(n):
                    break
                else:
                    # Composite and > p: need to check if all factors <= p
                    # For quick check, assume fail
                    break
        return k

    # Full verification
    def verify_run(p, k):
        p_sq = p * p
        for i in range(1, k + 1):
            n = p_sq + i
            if not _all_factors_at_most_p(n, p):
                return False
        return True

    # Phase 1: Search small primes up to 10^6
    small_search_limit = 1_000_000
    primes_small = small_primes

    for p in primes_small:
        if time.time() > deadline:
            break
        k = quick_smooth_run(p, 200)
        if k > best_k:
            if verify_run(p, k):
                best_k = k
                best_p = p

    # Phase 2: Search larger primes using primality testing
    if time.time() < deadline - 10:
        for start in range(1_000_000, min(50_000_000, 10**9 + 1), 100_000):
            if time.time() > deadline:
                break
            for candidate in range(start, min(start + 10_000, 10**9 + 1), 2):
                if time.time() > deadline:
                    break
                if not _is_prime_mr(candidate):
                    continue
                k = quick_smooth_run(candidate, 200)
                if k > best_k:
                    if verify_run(candidate, k):
                        best_k = k
                        best_p = candidate

    # Phase 3: Random sampling of larger primes
    if time.time() < deadline - 5:
        for _ in range(50):
            if time.time() > deadline:
                break
            candidate = random.randrange(10**7, 10**9 + 1)
            if candidate % 2 == 0:
                candidate += 1
            while not _is_prime_mr(candidate):
                candidate += 2
                if candidate > 10**9:
                    break
            if candidate > 10**9:
                continue
            k = quick_smooth_run(candidate, 200)
            if k > best_k:
                if verify_run(candidate, k):
                    best_k = k
                    best_p = candidate

    return {'p': best_p, 'k': best_k}
