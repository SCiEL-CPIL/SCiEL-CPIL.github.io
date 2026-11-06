# EVOLVE-BLOCK-START
def _totient(n: int) -> int:
    """Compute Euler's totient phi(n) via trial factorization."""
    result = n
    x = n
    d = 2
    while d * d <= x:
        if x % d == 0:
            while x % d == 0:
                x //= d
            result -= result // d
        d += 1
    if x > 1:
        result -= result // x
    return result


def construct():
    """
    Return the minimum n in [1, 1000000] such that phi(n) % 120 == 0.

    Strategy:
      1. Quick check of small n (1..200) to confirm the answer quickly.
      2. Number-theoretic pruning: for phi(n) to be divisible by 120 = 2^3 * 3 * 5,
         every prime p | n must have (p-1) contributing factors that collectively
         cover 2^3, 3, and 5. The only primes p with (p-1) | 120 are
         {2, 3, 5, 7, 11, 13, 31}.  We generate candidates from these primes
         (with exponent 1 or 2) and check phi(c) % 120 == 0.
      3. Fallback: linear scan from 1 upward if the pruned search misses something.
    """
    import time
    t0 = time.time()
    deadline = t0 + 60.0  # 60 s wall-clock safety margin

    # --- Phase 1: small-n brute force (covers the true minimum if it's small) ---
    for n in range(1, 201):
        if _totient(n) % 120 == 0:
            return n

    # --- Phase 2: number-theoretic candidate generation ---
    # Primes p where (p-1) divides 120: p-1 | 120
    # 120 = 2^3 * 3 * 5
    # p-1 can be any divisor of 120; p must be prime.
    divisors_of_120 = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120]
    candidate_primes = []
    for d in divisors_of_120:
        p = d + 1
        # Check primality
        if p < 2:
            continue
        is_prime = True
        for q in range(2, int(p ** 0.5) + 1):
            if p % q == 0:
                is_prime = False
                break
        if is_prime:
            candidate_primes.append(p)
    # candidate_primes should be [2, 3, 5, 7, 11, 13, 31]
    # (p-1 values: 1,2,4,6,10,12,30 — all divide 120)

    # Generate candidates as products of powers of these primes, each prime
    # used with exponent 1 or 2 (since p^3 for p>=3 gives p^2*(p-1) which
    # may still work, but we keep exponents small to stay under 1e6).
    # Actually, allow exponent 1..3 for p=2, and 1..2 for odd primes.
    from itertools import product

    exp_ranges = {2: range(0, 7), 3: range(0, 3), 5: range(0, 3),
                  7: range(0, 3), 11: range(0, 3), 13: range(0, 3),
                  31: range(0, 3)}
    primes_list = [2, 3, 5, 7, 11, 13, 31]
    exp_lists = [exp_ranges[p] for p in primes_list]

    best = 1000001
    for exps in product(*exp_lists):
        n = 1
        for p, e in zip(primes_list, exps):
            n *= p ** e
        if n > 1000000:
            continue
        if n < 2:
            continue
        if _totient(n) % 120 == 0 and n < best:
            best = n

    # --- Phase 3: fallback linear scan (only if pruned search didn't find anything
    #     smaller than what we already know; in practice Phase 1 already returned) ---
    if best > 200:
        for n in range(201, 1000001):
            if (time.time() - t0) > deadline - 5.0:
                break
            if _totient(n) % 120 == 0 and n < best:
                best = n
                break  # we scan upward, so first hit is the minimum in this range

    return best
# EVOLVE-BLOCK-END
