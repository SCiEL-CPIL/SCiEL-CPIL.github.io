import math

def construct():
    """
    Returns the largest subset of [1, 255] containing 255 such that 
    gcd(a, b) > 1 for all distinct pairs a, b in the subset.
    
    The maximum cardinality is 85, achieved by the set of all multiples of 3 
    in the range [1, 255]. 
    """
    n = 255
    # The smallest prime factor of 255 is 3.
    # The set of multiples of 3 is a valid pairwise non-coprime set.
    # Size is floor(255 / 3) = 85.
    # It can be proven that no larger set exists for n=255.
    
    # Generate the set of multiples of 3
    result = list(range(3, n + 1, 3))
    
    return result
