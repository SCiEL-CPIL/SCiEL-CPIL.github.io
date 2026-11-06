"""
Optimized construction for large pairwise noncoprime sets containing 255.

The problem asks for the largest subset S of {1, ..., 255} such that:
1. 255 is in S.
2. gcd(a, b) > 1 for all distinct a, b in S.

Analysis:
- 255 = 3 * 5 * 17.
- Any element in S must share a factor with 255, so it must be divisible by 3, 5, or 17.
- The set of all multiples of 3 in [1, 255] has size floor(255/3) = 85.
- This set is a valid clique because any two multiples of 3 have gcd at least 3.
- It contains 255.
- Can we do better?
  - If we include any number not divisible by 3, it must be divisible by 5 or 17.
  - Let S be a maximal clique.
  - If S contains a number x not divisible by 3, then for any y in S divisible by 3, gcd(x, y) > 1.
  - Since 3 does not divide x, the common factor must be 5 or 17 (or another prime, but restricted by the range).
  - This forces the multiples of 3 in S to be a subset of multiples of 15 or 51.
  - |Multiples of 15| = 17, |Multiples of 51| = 5. Union is 21.
  - So if we deviate from "all multiples of 3", we can keep at most 21 multiples of 3.
  - The non-multiples of 3 we add are from multiples of 5 (not 3) or 17 (not 3).
  - Max non-multiples of 3 compatible with each other is limited (e.g., all multiples of 5 not 3 is 34).
  - Total size would be at most 21 + 34 = 55, which is less than 85.
  - Thus, the set of all multiples of 3 is optimal.
"""
import math
import time

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # The optimal solution is the set of all multiples of 3 in [1, 255].
    # Size: 85
    # Contains 255: Yes (255 = 3 * 85)
    # Pairwise non-coprime: Yes (all divisible by 3)
    
    result = list(range(3, 256, 3))
    
    # Verification (fast enough for N=85)
    assert 255 in result
    assert len(result) == 85
    
    # Check time to ensure we are well within limits
    if time.time() > deadline:
        # This should never happen, but just in case
        return result

    return result

# EVOLVE-BLOCK-END
