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
    - Every element must be coprime-free with 255, so each is divisible by 3, 5, or 17.
    - If the set contains any element divisible by 5 but not 3, then every other element
      must also be divisible by 5 (since 5 is prime and must share a factor with it).
      This limits us to at most floor(255/5) = 51 elements.
    - Similarly for 17: at most floor(255/17) = 15.
    - The only way to exceed 51 is to have all elements divisible by 3, giving 85.
    - 85 is therefore the maximum cardinality.
    """
    start_time = time.time()
    deadline = 60.0
    
    # All multiples of 3 in [1, 255]: {3, 6, 9, ..., 255}
    # Size: floor(255/3) = 85
    result = list(range(3, 256, 3))
    
    # 255 = 3*85, so 255 % 3 == 0, confirming 255 is in result
    assert 255 in result, "255 must be in the result set"
    
    # Deadline guard (should never trigger for this O(1) computation)
    if time.time() - start_time > deadline:
        return [255]
    
    return result
# EVOLVE-BLOCK-END
