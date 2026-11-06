"""
Returns the set of all multiples of 3 in [1, 255].
This set contains 255.
For any two distinct elements a, b in this set, 3 divides both, so gcd(a, b) >= 3 > 1.
The size is 85, which is the maximum possible size for such a set.
"""
# EVOLVE-BLOCK-START
def construct():
    return list(range(3, 256, 3))
# EVOLVE-BLOCK-END
