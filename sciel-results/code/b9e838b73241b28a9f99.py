# EVOLVE-BLOCK-START
def _totient(n: int) -> int:
    """Compute Euler's totient phi(n) via trial factorization."""
    if n == 1:
        return 1
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

    The minimum such n is 143 (11 * 13), for which phi(143) = 120.
    We verify this by checking small n and generating candidates from
    primes p where p-1 divides 120.
    """
    # The known minimum is 143.
    # We perform a quick check to confirm and ensure no smaller n exists.
    # phi(n) must be a multiple of 120.
    # For n < 143, phi(n) < 143, so phi(n) must be 120.
    # We verify no n < 143 has phi(n) = 120.
    
    # Quick check for n in [1, 143]
    for n in range(1, 144):
        if _totient(n) % 120 == 0:
            return n

    # If for some reason 143 wasn't found (should not happen), 
    # fall back to the number-theoretic generation.
    # Primes p where (p-1) | 120
    candidate_primes = [2, 3, 5, 7, 11, 13, 31]
    
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
            
    return best
# EVOLVE-BLOCK-END
