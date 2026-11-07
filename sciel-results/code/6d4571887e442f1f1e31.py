"""Square-product extension for 210: find distinct ints in (210,10210]
whose product * 210 is a perfect square, minimizing max - 210."""

import math
import time
from collections import defaultdict

N = 210
MAX_N = 10210
MAX_K = 100
SMALL_PRIMES = [2, 3, 5, 7]


def _is_square(n):
    r = math.isqrt(n)
    return r * r == n


def _sieve_primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i in range(2, limit + 1) if sieve[i]]


def _factorize(n, primes):
    d = {}
    for p in primes:
        if p * p > n:
            break
        if n % p == 0:
            e = 0
            while n % p == 0:
                n //= p
                e += 1
            d[p] = e
    if n > 1:
        d[n] = 1
    return d


def construct():
    t_start = time.time()
    DEADLINE = t_start + 80.0

    primes = _sieve_primes(MAX_N)
    
    # Precompute parity for all candidates
    parity = {}
    for n in range(211, MAX_N + 1):
        d = _factorize(n, primes)
        mask = 0
        for i, p in enumerate(SMALL_PRIMES):
            if d.get(p, 0) & 1:
                mask |= (1 << i)
        large = frozenset(p for p, e in d.items() if p not in SMALL_PRIMES and e & 1)
        parity[n] = (mask, large)

    TARGET_MASK = 0b1111
    TARGET_LARGE = frozenset()

    best_val = None
    best_set = None

    # Check 1-element solutions
    for n in range(211, MAX_N + 1):
        if time.time() > DEADLINE:
            break
        m, lg = parity[n]
        if m == TARGET_MASK and lg == TARGET_LARGE:
            val = n - 210
            if best_val is None or val < best_val:
                best_val = val
                best_set = (n,)

    # Check 2-element solutions by grouping by large-prime set
    by_large = defaultdict(list)
    for n in range(211, MAX_N + 1):
        m, lg = parity[n]
        by_large[lg].append((m, n))

    for lg, lst in by_large.items():
        if time.time() > DEADLINE:
            break
        mask_to_nums = defaultdict(list)
        for m, n in lst:
            mask_to_nums[m].append(n)
        
        for m, nums in mask_to_nums.items():
            if time.time() > DEADLINE:
                break
            target_m = m ^ TARGET_MASK
            if target_m in mask_to_nums:
                other_nums = mask_to_nums[target_m]
                # Find pair with minimum max
                for a in sorted(nums):
                    for b in sorted(other_nums):
                        if a != b:
                            val = max(a, b) - 210
                            if best_val is None or val < best_val:
                                best_val = val
                                best_set = (a, b)
                            break
                    break
                break

    # If we found a good solution, validate and return
    if best_set is not None:
        prod = N
        for x in best_set:
            prod *= x
        if _is_square(prod) and len(set(best_set)) == len(best_set) and all(
            isinstance(x, int) and not isinstance(x, bool) and 210 < x <= MAX_N for x in best_set
        ):
            return [int(x) for x in best_set]

    # Fallback
    return [840]


if __name__ == "__main__":
    t0 = time.time()
    result = construct()
    t1 = time.time()
    print(f"Result: {result}")
    print(f"Max - 210: {max(result) - 210}")
    prod = 210
    for x in result:
        prod *= x
    print(f"Product is square: {_is_square(prod)}")
    print(f"Time: {t1 - t0:.2f}s")
