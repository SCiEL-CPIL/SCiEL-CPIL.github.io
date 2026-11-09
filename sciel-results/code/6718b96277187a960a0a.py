import time

def construct():
    start = time.time()
    deadline = start + 50.0  # safety margin before 60s limit

    # The seed returns 12 primes in 2..40.
    # We know we can do better using prime powers with distinct subset products.
    #
    # Strategy: Use primes and prime powers. For distinct subset products,
    # we need that no non-empty multiset of selected numbers can have the same
    # prime factorization as another non-empty multiset.
    #
    # The known optimal approach: select primes and some prime powers such that
    # the exponents form a set with distinct subset sums (Sidon-like property).
    #
    # For primes in 2..40: 2,3,5,7,11,13,17,19,23,29,31,37 (12 primes)
    # We can add prime powers: 4=2^2, 8=2^3, 9=3^2, 16=2^4, 25=5^2, 27=3^3, 32=2^5
    #
    # Key insight: If we pick numbers whose prime factorizations are "independent"
    # in the sense that the exponent vectors form a set with distinct subset sums,
    # then all subset products are distinct.
    #
    # Let's construct a set of 16:
    # - For prime 2: pick 2, 4, 16 (exponents 1, 2, 4) -> subset sums of {1,2,4} are distinct
    # - For prime 3: pick 3, 9 (exponents 1, 2) -> subset sums distinct
    # - For prime 5: pick 5, 25 (exponents 1, 2) -> subset sums distinct
    # - For primes 7,11,13,17,19,23,29,31,37: pick each prime itself (9 more)
    # Total: 3 + 2 + 2 + 9 = 16
    #
    # All these numbers are in 2..40 and are distinct.
    # All subset products are distinct because each prime's contribution to any subset
    # product is determined by a unique sum of exponents from its selected powers.

    candidates = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    # Verify distinctness of all subset products
    # For k=16, there are 2^16 = 65536 subset products. We can enumerate them.
    products = set()
    for mask in range(1 << len(candidates)):
        prod = 1
        for i, c in enumerate(candidates):
            if mask & (1 << i):
                prod *= c
        if prod in products:
            # Collision found - should not happen with our construction
            # Fall back to smaller set
            break
        products.add(prod)
    else:
        # All subset products are distinct
        return list(candidates)

    # Fallback: return the 12 primes
    return [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
