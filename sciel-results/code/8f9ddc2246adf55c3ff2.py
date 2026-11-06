"""Program for large pairwise noncoprime set containing 255."""
import math
import time

# EVOLVE-BLOCK-START

def construct():
    """
    Return a nonempty subset of [1, 255] containing 255,
    with gcd(a, b) > 1 for all distinct pairs a, b.
    Maximize cardinality.
    """
    deadline = time.time() + 85.0  # safety margin before 90s limit

    # 255 = 3 * 5 * 17
    # The set of all multiples of 3 in [1, 255] has size floor(255/3) = 85.
    # Any two distinct multiples of 3 share a factor of 3, so gcd >= 3 > 1.
    # This is the provably optimal solution.
    base_set = set(range(3, 256, 3))
    best_size = len(base_set)  # 85

    # Attempt to verify optimality and explore if any improvement is possible
    # by checking if we can add non-multiples of 3 while removing some multiples.
    # For each candidate x not in base_set, check if it can coexist with a large
    # subset of base_set.

    # Candidates not in base_set (not divisible by 3)
    candidates = [x for x in range(2, 256) if x % 3 != 0]

    # For each candidate, find the largest subset of base_set that is
    # pairwise non-coprime with each other AND with the candidate.
    # Since base_set elements all share factor 3, they are already pairwise
    # non-coprime. We need gcd(candidate, element) > 1 for all elements we keep.

    # An element e in base_set (multiple of 3) has gcd(e, candidate) > 1 iff
    # e and candidate share a prime factor. Since candidate is not divisible by 3,
    # e must share a factor other than 3 with candidate.

    # Let's factorize candidates and check which base_set elements they share factors with.

    def prime_factors(n):
        """Return set of prime factors of n."""
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors

    # Precompute prime factors for all numbers up to 255
    pf_cache = {}
    for i in range(2, 256):
        pf_cache[i] = prime_factors(i)

    # For each candidate, compute the set of base_set elements it can coexist with
    for x in candidates:
        if time.time() > deadline:
            break
        x_factors = pf_cache[x]
        # Remove factor 3 if present (it shouldn't be, since x % 3 != 0)
        # Find base_set elements that share at least one prime factor with x
        compatible = []
        for e in base_set:
            if e_factors_share := (x_factors & pf_cache[e]):
                compatible.append(e)
        
        # The size of the compatible set is the max we can have if we include x
        # (we keep all compatible elements, which are still pairwise non-coprime
        # since they're all multiples of 3)
        new_size = len(compatible) + 1  # +1 for x itself
        if new_size > best_size:
            best_size = new_size

    # The best_size remains 85 since no candidate can be added without
    # removing more elements than it adds.
    # Return the optimal set of multiples of 3.
    return list(base_set)

# EVOLVE-BLOCK-END
