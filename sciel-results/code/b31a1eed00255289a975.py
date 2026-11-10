import time
import random

def construct():
    # The minimal largest element for n distinct positive integers with distinct subset sums
    # is 2^(n-1). The set is {1, 2, 4, ..., 2^(n-1)}.
    # This is because the subset sums must be distinct, implying the sequence must be
    # superincreasing (each element greater than the sum of all previous elements)
    # to avoid collisions in the subset sums.
    # The minimal superincreasing sequence starting with 1 is powers of 2.
    
    n = 12
    result = [1 << i for i in range(n)]
    return result
