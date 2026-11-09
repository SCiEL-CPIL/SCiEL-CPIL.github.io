"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def construct():
    """
    Return a nonempty subset of [1,255] containing 255, with gcd(a,b)>1 for all distinct pairs.
    Maximize cardinality.
    
    The maximum size is 85, achieved by all multiples of 3 in [1, 255].
    255 is a multiple of 3.
    For any x, y multiples of 3, gcd(x, y) >= 3 > 1.
    Any set containing a multiple of 3 that is not a multiple of 5 or 17
    must be a subset of multiples of 3.
    Sets not containing such elements are subsets of multiples of 5 or 17,
    which have smaller cardinalities (51 and 15).
    """
    return [i for i in range(1, 256) if i % 3 == 0]

# EVOLVE-BLOCK-END
