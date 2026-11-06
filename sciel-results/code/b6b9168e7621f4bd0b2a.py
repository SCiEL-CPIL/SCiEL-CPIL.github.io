import math
import time

def construct():
    # The optimal strategy for n=255 (3*5*17) is to take all multiples of the smallest prime factor, 3.
    # This yields a set of size floor(255/3) = 85.
    # This is the maximum possible size for a pairwise non-coprime subset containing 255.
    # Any set containing a number not divisible by 3 must exclude all multiples of 3 that are coprime to it,
    # which results in a net loss of cardinality.
    
    # Generate the sorted list of multiples of 3 in [1, 255]
    # range(3, 256, 3) generates 3, 6, ..., 255
    result = list(range(3, 256, 3))
    
    return result
