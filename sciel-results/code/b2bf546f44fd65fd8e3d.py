import time
import math
from fractions import Fraction

MAX = 10000000
TARGET = Fraction(3, 2)
DEADLINE = time.time() + 78.0   # wall-clock safety margin (< 90s)
ODD_LIMIT = 3400000

def construct():
    start = time.time()

    # --- Precompute sigma for all odd m up to ODD_LIMIT (sieve) ---
    odd_lim = ODD_LIMIT
    sigma_odd = [1] * (odd_lim + 1)
    for p in range(3, odd_lim + 1, 2):
        if sigma_odd[p] == 1:           # p is prime
            for k in range(p, odd_lim + 1, p):
                pk = p
                term = p + 1
                while k * p <= odd_lim:
                    k *= p
                    pk *= p
                    term += pk
                sigma_odd[k] *= term

    def sigma(n):
        # sigma(n) = (2^(a+1)-1)*sigma(m), n = 2^a * m (m odd)
        if n <= odd_lim:
            m = n >> (n & -n).bit_length() - 1  # not used; do explicit below
        a = 0
        m = n
        while m & 1 == 0:
            m >>= 1
            a += 1
        if m <= odd_lim:
            return ((1 << (a + 1)) - 1) * sigma_odd[m]
        # fallback exact factorization (rare)
        x = n
        s = 1
        d = 2
        while d * d <= x:
            if x % d == 0:
                e = 0
                while x % d == 0:
                    x //= d
                    e += 1
                s *= (d ** (e + 1) - 1) // (d - 1)
            d += 1 if d == 2 else 2
        if x > 1:
            s *= (x + 1)
        return s

    best_err = None
    best = (11, 6)

    # --- Scan b; test a = round(3b/2) for sigma(a)==sigma(b) ---
    for b in range(1, MAX + 1):
        if time.time() > DEADLINE:
            break
        sa = None
        # candidate a values near 3b/2
        base = (3 * b) // 2
        for a in (base, base + 1):
            if a < 1 or a > MAX:
                continue
            if sa is None:
                sa = sigma(a)
            if sa == sigma(b):
                f = Fraction(a, b)
                if f == TARGET:
                    continue  # raw value exactly 0 is invalid
                err = abs(f - TARGET)
                if best_err is None or err < best_err:
                    best_err = err
                    best = (a, b)

    if best_err is None:
        # fallback to the seed pair (guaranteed valid)
        return [11, 6]
    return [int(best[0]), int(best[1])]
