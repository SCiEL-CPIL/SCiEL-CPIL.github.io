import math

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a nonempty subset of [1, 255] containing 255, with gcd(a, b) > 1 
    for all distinct pairs. Maximizes cardinality.
    
    255 = 3 * 5 * 17.
    The largest pairwise non-coprime subset containing 255 consists of all 
    multiples of the smallest prime factor of 255, which is 3.
    """
    # Multiples of 3 in [1, 255]
    # 3, 6, ..., 255
    return list(range(3, 256, 3))
# EVOLVE-BLOCK-END
