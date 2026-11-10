"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    """
    Return a nonempty subset of [1, 255] containing 255,
    with gcd(a,b)>1 for all distinct pairs. Maximize cardinality.
    
    255 = 3 * 5 * 17.
    
    Optimal strategy: return all multiples of 3 in [1, 255].
    - 255 is a multiple of 3.
    - Any two multiples of 3 share the factor 3, so gcd(a,b) >= 3 > 1.
    - The size is floor(255/3) = 85.
    
    Proof of optimality:
    - Every element must share a factor with 255, so each is divisible by 3, 5, or 17.
    - If the set contains any element divisible by 5 but not 3, then every other element
      must also share a factor with it. Any multiple of 3 not divisible by 5 would have
      gcd 1 with such an element. So all multiples of 3 kept must also be divisible by 5,
      i.e., multiples of 15 (only 17 such elements). Adding even one 5-divisible non-3
      element drops the max achievable well below 85.
    - Similarly for 17-divisible non-3 elements.
    - Multiples of 3 alone give 85, which is the maximum.
    - Multiples of 5 give 51, multiples of 17 give 15 — both smaller.
    """
    start_time = time.time()
    deadline = 55.0  # Safety margin below the 90s limit
    
    # All multiples of 3 in [1, 255]: {3, 6, 9, ..., 255}
    # Size: floor(255/3) = 85
    result = list(range(3, 256, 3))
    
    # Verify 255 is in the set (255 = 3*85, so yes)
    assert 255 in result, "255 must be in the result set"
    
    # Quick pairwise gcd verification for correctness
    # n=85, so 85*84/2 = 3570 pairs — trivially fast
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            if math.gcd(result[i], result[j]) <= 1:
                # Should never happen; fall back to safe answer
                return [255]
    
    # Deadline guard (should never trigger for this O(1) computation)
    if time.time() - start_time > deadline:
        return [255]
    
    return result
# EVOLVE-BLOCK-END
