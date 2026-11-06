"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    """
    Return a nonempty subset of [1,255] containing 255, with gcd(a,b)>1 for all distinct pairs.
    Maximize cardinality.
    
    255 = 3 * 5 * 17.
    The largest pairwise non-coprime set containing 255 is obtained by taking all multiples
    of the smallest prime factor of 255, which is 3.
    
    All multiples of 3 in [1, 255]: 3, 6, 9, ..., 255.
    Count: 255 // 3 = 85 elements.
    
    This is optimal because:
    - Any element in the set must share a common factor with 255, so each element must be
      divisible by 3, 5, or 17.
    - If we take elements from different prime factors, they must still be pairwise non-coprime.
    - The largest such set is achieved by taking all multiples of the smallest prime factor (3),
      giving 85 elements.
    """
    # All multiples of 3 from 3 to 255
    result = list(range(3, 256, 3))
    return result
# EVOLVE-BLOCK-END
