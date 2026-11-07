"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # Each integer 2..40 is represented by its prime-exponent vector.
    # Two subset products are distinct iff the multiset of exponent vectors
    # is such that all 2^k integer linear combinations with coefficients in {0,1}
    # are distinct.  This is equivalent to the vectors being linearly independent
    # over Z with coefficients restricted to {-1,0,1} in the difference sense,
    # but more simply: no subset of the vectors sums (with +/-1 coefficients)
    # to the zero vector.
    #
    # We build the set greedily by considering candidates in an order that
    # maximizes independence, then verify with a full subset-product enumeration.

    from math import isqrt
    from itertools import combinations

    # Factor each number 2..40 into prime-exponent tuple.
    def factor(n):
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return factors

    # Build exponent vectors for 2..40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    vecs = {}
    for n in range(2, 41):
        f = factor(n)
        vecs[n] = tuple(f.get(p, 0) for p in primes)

    # Greedy: try to add candidates in order of "usefulness".
    # We'll try all subsets via a smarter approach: since max size is at most
    # the dimension of the rational span (which is 12 here), we can do a
    # bounded search. But 40 choose 12 is too large. Instead, use greedy
    # with verification.

    # Strategy: iterate candidates, try to add if the resulting set still has
    # distinct subset products. Use randomized/greedy ordering for speed.

    def has_distinct_subset_products(selected):
        """Check that all 2^len(selected) subset products are distinct."""
        k = len(selected)
        if k > 20:
            return False  # safety
        products = set()
        for mask in range(1 << k):
            prod = 1
            for i in range(k):
                if mask >> i & 1:
                    prod *= selected[i]
            if prod in products:
                return False
            products.add(prod)
        return True

    # Try to build a large set using greedy addition with backtracking for a few steps.
    # First, get a strong initial set using the prime-power decomposition insight.
    # Then try to improve.

    # Strong initial set based on prime powers:
    # Powers of 2 up to 40: 2,4,8,16,32 -> pick {4,8,16} (max independent subset of size 3)
    # Powers of 3: 3,9,27 -> pick {3,9}
    # Powers of 5: 5,25 -> pick {5,25}
    # Powers of 7: 7 -> pick {7}
    # Remaining primes: 11,13,17,19,23,29,31,37 -> one each
    initial = [4, 8, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    # Verify initial set
    if not has_distinct_subset_products(initial):
        # Fallback: just use primes
        initial = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    # Try to add more elements
    remaining = [n for n in range(2, 41) if n not in initial]
    # Sort remaining by some heuristic: prefer numbers with unique prime factors
    # or numbers that are "new" in some sense. Just try in order.
    for n in remaining:
        trial = initial + [n]
        if has_distinct_subset_products(trial):
            initial = trial

    # Try another pass in reverse order to catch different combos
    for n in sorted(remaining, reverse=True):
        if n in initial:
            continue
        trial = initial + [n]
        if has_distinct_subset_products(trial):
            initial = trial

    return list(initial)

# EVOLVE-BLOCK-END
