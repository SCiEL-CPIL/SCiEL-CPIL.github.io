import time
import numpy as np

N = 100
DEADLINE_SECONDS = 145.0

# EVOLVE-BLOCK-START
def construct():
    """
    OneMax: maximize the number of 1s in a binary string of length N.
    Direct construction: the optimal solution is all ones.
    """
    # Directly construct the string of N '1's
    # This is the optimal solution and requires no computation or time tracking
    # for correctness, but we keep the structure minimal.
    return '1' * N
# EVOLVE-BLOCK-END
