"""
Constructs a nonempty subset of [1, 255] containing 255,
with gcd(a, b) > 1 for all distinct pairs, maximizing cardinality.

For n = 255 = 3 * 5 * 17, the largest pairwise non-coprime set
containing 255 is the set of all multiples of the smallest prime factor, 3.
Any two distinct multiples of 3 have a gcd of at least 3.
Size of this set is floor(255 / 3) = 85.
"""

def construct():
    # The prime factors of 255 are 3, 5, and 17.
    # To maximize the size of the set {k | gcd(k, m) > 1 for all m in set},
    # we choose the multiples of the smallest prime factor.
    # Multiples of 3: 3, 6, ..., 255. Count = 85.
    # Multiples of 5: 5, 10, ..., 255. Count = 51.
    # Multiples of 17: 17, 34, ..., 255. Count = 15.
    
    # Returning multiples of 3 gives the maximum cardinality of 85.
    return list(range(3, 256, 3))
