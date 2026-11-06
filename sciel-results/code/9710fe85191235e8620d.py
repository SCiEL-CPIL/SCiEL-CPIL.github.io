import time
import random
import math

# ── Deterministic Miller-Rabin for n < 2^64 ──────────────────────────────────
_MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37,
                 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
                 101, 103, 107, 109, 113, 127, 131, 137, 139, 149,
                 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]

def is_prime(n):
    """Deterministic Miller-Rabin.  Exact for n < 2^64 with 12 base primes."""
    if n < 2:
        return False
    for p in _SMALL_PRIMES:
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

# ── Pollard's Rho / Brent factorisation ──────────────────────────────────────

def _pollard_brent(n):
    """Return a non-trivial factor of composite n."""
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    if n % 5 == 0:
        return 5
    c = random.randrange(1, n - 1)
    y = random.randrange(0, n - 1)
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
    if g == n:
        return _pollard_brent(n)
    return g

def _factor(n, factors):
    """Recursively factor n, appending prime factors to *factors*."""
    if n == 1:
        return
    for p in _SMALL_PRIMES:
        if n % p == 0:
            factors.append(p)
            _factor(n // p, factors)
            return
    if is_prime(n):
        factors.append(n)
        return
    d = _pollard_brent(n)
    _factor(d, factors)
    _factor(n // d, factors)

def all_prime_factors_leq(n, limit):
    """True iff every prime factor of n is ≤ limit.  Early-exit on violation."""
    if n <= 1:
        return True
    for p in _SMALL_PRIMES:
        if p > limit:
            break
        if n % p == 0:
            while n % p == 0:
                n //= p
            if n == 1:
                return True
    if n == 1:
        return True
    if n <= limit:
        return True
    if n > limit and is_prime(n):
        return False
    flag = [True]
    def _factor_bounded(m):
        if m == 1:
            return
        for p in _SMALL_PRIMES:
            if p > limit:
                break
            if m % p == 0:
                _factor_bounded(m // p)
                return
        if m <= limit:
            return
        if is_prime(m):
            if m > limit:
                flag[0] = False
            return
        d = _pollard_brent(m)
        _factor_bounded(d)
        if flag[0]:
            _factor_bounded(m // d)
    _factor_bounded(n)
    return flag[0]

# ── Sieve ─────────────────────────────────────────────────────────────────────

def sieve_upto(limit):
    """Return a boolean bytearray where index i marks composites up to limit."""
    size = limit + 1
    comp = bytearray(size)
    for i in range(2, int(math.isqrt(limit)) + 1):
        if comp[i] == 0:
            start_sq = i * i
            comp[start_sq::i] = b'\x01' * ((limit - start_sq) // i + 1)
    return comp

# ── Main construct() ─────────────────────────────────────────────────────────

def construct():
    start = time.time()
    deadline = start + 140   # 10 s safety margin before 150 s wall-clock kill

    best_k = 0
    best_p = 2

    # Phase 1: sieve small primes up to 10^7, check each
    SIEVE_LIMIT = 10_000_000
    comp = sieve_upto(SIEVE_LIMIT)

    for n_candidate in range(2, SIEVE_LIMIT + 1):
        if time.time() > deadline:
            break
        if comp[n_candidate] == 0:  # prime
            p = n_candidate
            base = p * p
            k = 0
            i = 1
            while i <= 200:
                if time.time() > deadline:
                    break
                val = base + i
                if not all_prime_factors_leq(val, p):
                    break
                k += 1
                i += 1
            if k > best_k:
                best_k = k
                best_p = p

    # Phase 2: random primes in (SIEVE_LIMIT, 10^9) using Miller-Rabin
    # Focus on larger p since larger p allows more room for p-smooth p^2+i
    random.seed(42)
    tried = set()
    attempts = 0
    max_attempts = 500000

    while attempts < max_attempts:
        if time.time() > deadline:
            break
        attempts += 1
        # Bias toward larger primes
        if random.random() < 0.6:
            nc = random.randint(500_000_000, 1_000_000_000)
        elif random.random() < 0.5:
            nc = random.randint(100_000_000, 500_000_000)
        else:
            nc = random.randint(SIEVE_LIMIT + 1, 100_000_000)
        if nc <= 2 or nc > 10**9:
            continue
        if nc in tried:
            continue
        if nc % 2 == 0 and nc > 2:
            nc -= 1
        tried.add(nc)
        if not is_prime(nc):
            continue
        p = nc
        base = p * p
        k = 0
        i = 1
        while i <= 200:
            if time.time() > deadline:
                break
            val = base + i
            if not all_prime_factors_leq(val, p):
                break
            k += 1
            i += 1
        if k > best_k:
            best_k = k
            best_p = p

    # Phase 3: directed search near 10^9 (largest primes give most room)
    for offset in range(0, 500_000, 2):
        if time.time() > deadline:
            break
        c = 1_000_000_000 - offset
        if c % 2 == 0 and c > 2:
            c -= 1
        if c <= SIEVE_LIMIT:
            break
        if c in tried:
            continue
        tried.add(c)
        if not is_prime(c):
            continue
        p = c
        base = p * p
        k = 0
        i = 1
        while i <= 200:
            if time.time() > deadline:
                break
            val = base + i
            if not all_prime_factors_leq(val, p):
                break
            k += 1
            i += 1
        if k > best_k:
            best_k = k
            best_p = p

    return {'p': best_p, 'k': best_k}
