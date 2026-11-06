import time
import math

DENOM_BOUND = 10**12


def _factorize(m):
    """Return list of (prime, exponent) for m >= 1."""
    factors = []
    if m <= 1:
        return factors
    c = 0
    while m % 2 == 0:
        m //= 2
        c += 1
    if c:
        factors.append((2, c))
    f = 3
    while f * f <= m:
        if m % f == 0:
            c = 0
            while m % f == 0:
                m //= f
                c += 1
            factors.append((f, c))
        f += 2
    if m > 1:
        factors.append((m, 1))
    return factors


def _gen_divisors(factors):
    """Generate all divisors from a list of (prime, exponent)."""
    divs = [1]
    for p, e in factors:
        cur = list(divs)
        v = 1
        for _ in range(e):
            v *= p
            for d in cur:
                divs.append(d * v)
    return divs


def _solve_one(n, deadline):
    # x must be > n/4 (so that 1/x < 4/n)
    x_min = n // 4 + 1
    # If x > n, then 1/x < 1/n, so 1/y+1/z < 3/n < 4/n, impossible.
    x_max = n
    for x in range(x_min, x_max + 1):
        A = 4 * x - n
        B = n * x
        B2 = B * B
        lo = B + 1
        hi = B  # valid t in (0, B)
        if lo > hi:
            continue
        # divisors of B^2
        divs = _gen_divisors(_factorize(B2))
        for t in divs:
            if t < lo:
                continue
            if t > hi:
                break  # divisors are sorted ascending
            y = (B + t) // A
            z = (B + t + A * A) // A
            # distinctness and ordering
            if y <= x or z <= y or z > DENOM_BOUND:
                continue
            # verify exactness via cross multiplication
            if n * (y * z + x * z + x * y) == 4 * x * y * z:
                return [x, y, z]
        if time.time() > deadline:
            break
    return None


def construct():
    start = time.time()
    deadline = start + 85.0
    results = [None] * 254

    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        results[idx] = _solve_one(n, deadline)

    return results
