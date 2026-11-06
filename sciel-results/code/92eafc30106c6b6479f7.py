"""Search for n in [10000,1000000] maximizing powerful_part(n)*powerful_part(n+1)."""
import time
import math

# EVOLVE-BLOCK-START

def _build_spf(limit):
    """Build smallest prime factor array up to limit."""
    spf = list(range(limit + 1))
    if limit >= 0:
        spf[0] = 0
    if limit >= 1:
        spf[1] = 1
    for i in range(2, int(math.isqrt(limit)) + 1):
        if spf[i] == i:
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf


def _compute_powerful_parts(n_max):
    """
    Compute powerful part for every number 1..n_max.
    powerful_part(m) = product of p^e for each prime p with exponent e>=2 in factorization of m.
    Uses SPF for fast factorization.
    """
    spf = _build_spf(n_max)
    pp = [1] * (n_max + 1)
    for m in range(2, n_max + 1):
        temp = m
        prod = 1
        while temp > 1:
            p = spf[temp]
            e = 0
            while temp % p == 0:
                temp //= p
                e += 1
            if e >= 2:
                prod *= (p ** e)
        pp[m] = prod
    return pp


def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin before 90s kill

    n_min = 10000
    n_max = 1000000

    # Precompute powerful parts for all numbers up to n_max
    pp = _compute_powerful_parts(n_max)

    # Search for the n in [n_min, n_max-1] that maximizes pp[n]*pp[n+1]
    best_n = n_min
    best_val = pp[n_min] * pp[n_min + 1]

    for n in range(n_min, n_max):
        val = pp[n] * pp[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
        if time.time() > deadline:
            break

    return best_n

# EVOLVE-BLOCK-END
