"""Rewritten: fiber-based search for sigma-equal pairs minimizing |a/b - 3/2|."""
import time
import bisect
from collections import defaultdict

# EVOLVE-BLOCK-START
N = 10_000_000

def _linear_sigma(n):
    """Compute sigma(1..n) using linear sieve."""
    sigma = [0] * (n + 1)
    spow = [0] * (n + 1)    # spow[i] = p^k where p is smallest prime factor, k maximal
    spsum = [0] * (n + 1)   # spsum[i] = 1 + p + p^2 + ... + p^k
    primes = []
    is_comp = [False] * (n + 1)

    sigma[1] = 1
    spow[1] = 1
    spsum[1] = 1

    for i in range(2, n + 1):
        if not is_comp[i]:
            primes.append(i)
            spow[i] = i
            spsum[i] = 1 + i
            sigma[i] = 1 + i
        for p in primes:
            prod = i * p
            if prod > n:
                break
            is_comp[prod] = True
            if i % p == 0:
                spow[prod] = spow[i] * p
                spsum[prod] = spsum[i] + spow[prod]
                sigma[prod] = sigma[i] // spsum[i] * spsum[prod]
                break
            else:
                spow[prod] = p
                spsum[prod] = 1 + p
                sigma[prod] = sigma[i] * (1 + p)

    return sigma

def construct():
    t0 = time.time()
    DEADLINE = t0 + 78.0

    sigma = _linear_sigma(N)

    fibers = defaultdict(list)
    for n in range(1, N + 1):
        fibers[sigma[n]].append(n)

    best_a, best_b = 11, 6
    best_num = abs(2*11 - 3*6)  # 4
    best_den = 2*6              # 12

    for s, lst in fibers.items():
        if time.time() > DEADLINE:
            break
        ln = len(lst)
        if ln < 2:
            continue
        for i in range(ln):
            if time.time() > DEADLINE:
                break
            b = lst[i]
            j = bisect.bisect_left(lst, (3 * b) // 2)
            for k in (j, j - 1):
                if 0 <= k < ln and k != i:
                    a = lst[k]
                    num = abs(2*a - 3*b)
                    den = 2*b
                    if num * best_den < best_num * den:
                        best_num = num
                        best_den = den
                        best_a, best_b = a, b

    if sigma[best_a] != sigma[best_b] or best_a == best_b:
        best_a, best_b = 11, 6

    return [best_a, best_b]
# EVOLVE-BLOCK-END
