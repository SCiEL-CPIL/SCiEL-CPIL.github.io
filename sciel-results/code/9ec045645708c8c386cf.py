"""
Constructs a maximum-cardinality subset of [1, 255] containing 255
where every distinct pair has gcd > 1.

255 = 3 * 5 * 17. The optimal strategy is to take all multiples of the
smallest prime factor (3), yielding 85 elements. Mixing primes from different
factor groups creates coprime pairs, so a single-prime-factor set is optimal.
"""

import time
from math import gcd

def construct():
    """
    Returns a nonempty subset of [1, 255] containing 255, with gcd(a,b) > 1
    for all distinct pairs, maximizing cardinality.
    """
    # All multiples of 3 in [1, 255]: size = 85
    # This is valid because gcd(3a, 3b) >= 3 > 1 for all a, b.
    # 255 is a multiple of 3, so it is included.
    result = list(range(3, 256, 3))
    return result
