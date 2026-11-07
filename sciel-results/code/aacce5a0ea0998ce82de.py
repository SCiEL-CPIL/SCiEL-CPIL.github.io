import math
import time

def _sigma(n):
    """Compute sum of divisors sigma(n) via trial division."""
    if n == 0:
        return 0
    result = 1
    m = n
    d = 2
    while d * d <= m:
        if m % d == 0:
            s = 1
            p = 1
            while m % d == 0:
                m //= d
                p *= d
                s += p
            result *= s
        d += 1 if d == 2 else 2
    if m > 1:
        result *= (1 + m)
    return result

def construct():
    deadline = time.time() + 140.0
    best = {'M': 333312, 'k': [336, 372, 434]}
    best_count = 3

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    pcount = len(primes)

    # Generate all numbers <= LIMIT composed only of the given primes
    LIMIT = 5000000
    smooth = []

    def gen(idx, val):
        if time.time() > deadline - 10.0:
            return
        if val > LIMIT:
            return
        if val > 1:
            smooth.append(val)
        for j in range(idx, pcount):
            p = primes[j]
            if val * p > LIMIT:
                break
            gen(j, val * p)

    gen(0, 1)

    # Group by k*sigma(k)
    groups = {}
    for k in smooth:
        s = _sigma(k)
        M = k * s
        if M <= 10**18:
            if M not in groups:
                groups[M] = []
            groups[M].append(k)

    # Find the M with the most preimages
    for M, ks in groups.items():
        if len(ks) > best_count:
            best_count = len(ks)
            best = {'M': M, 'k': ks[:1000]}

    # Verify the best result
    if best_count > 3:
        for k in best['k']:
            assert k * _sigma(k) == best['M'], f"Verification failed for k={k}"
            assert 1 <= k <= math.isqrt(best['M']), f"k={k} out of range"

    return {'M': int(best['M']), 'k': [int(x) for x in best['k']]}
