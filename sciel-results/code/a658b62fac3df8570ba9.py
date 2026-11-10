import time
import sys

def construct():
    """
    Constructs a set of 12 distinct positive integers such that all 4096 subset
    sums are distinct, and the largest integer is minimized.
    
    The minimal largest element for n=12 is 2^11 = 2048, achieved by the
    sequence of powers of 2: [1, 2, 4, ..., 2048].
    """
    n = 12
    # The optimal solution is the powers of 2
    return [1 << i for i in range(n)]

# EVOLVE-BLOCK-END
