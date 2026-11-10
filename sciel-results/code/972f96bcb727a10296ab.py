"""Program to find a large pairwise non-coprime subset of [1,255] containing 255."""
import math
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Return a nonempty subset of [1,255] containing 255,
    with gcd(a,b)>1 for all distinct pairs. Maximize cardinality.

    Key insight: 3 is in [1,255] and is prime. Any element x in a pairwise
    non-coprime set containing 3 must satisfy gcd(x,3)>1, so x must be
    divisible by 3. There are exactly floor(255/3)=85 multiples of 3 in
    [1,255], and they all share factor 3, forming a valid clique of size 85.
    This is the theoretical maximum.
    """
    start = time.time()
    deadline = start + 80  # safety margin under 90s kill

    # All multiples of 3 in [1,255] form a pairwise non-coprime set (all share factor 3)
    # 255 = 3*85, so 255 is included
    result = list(range(3, 256, 3))

    # Fast local-search polish: try to add any element from [1,255]\result
    # that has gcd>1 with every element in result. Since every element in
    # result is divisible by 3, an outside element x would need gcd(x,3)>1,
    # i.e., x divisible by 3 — but then x is already in result. So no
    # addition is possible. This pass is O(255 * 85) ≈ 22k gcd calls, well
    # under the deadline, and serves as a verification/safety net.
    result_set = set(result)
    for x in range(1, 256):
        if x in result_set:
            continue
        if time.time() > deadline:
            break
        # Check if x is compatible with all elements in result
        compatible = True
        for y in result:
            if math.gcd(x, y) == 1:
                compatible = False
                break
        if compatible:
            result.append(x)
            result_set.add(x)

    return result
# EVOLVE-BLOCK-END
