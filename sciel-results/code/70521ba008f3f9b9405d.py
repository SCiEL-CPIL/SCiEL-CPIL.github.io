"""
Optimized program for OneMax task.
Constructs a 100-bit binary string maximizing the number of ones.
"""
import time
import numpy as np

N = 100
DEADLINE_SECONDS = 150.0
SAFETY_MARGIN_SECONDS = 5.0

# Precompute the optimal solution once at module load time.
# This is the direct construction: all bits set to '1'.
# Using NumPy pre-allocation as per inspiration 3 and cell findings 
# to ensure constant time execution with zero branching in construct().
_OPTIMAL = ''.join(np.ones(N, dtype='U1'))


def construct():
    """
    Return a binary string of length N maximizing the number of ones.

    For the OneMax problem with N=100, the global optimum is the string
    of 100 ones. This function returns that precomputed string.
    
    The function executes in constant time with zero branching logic,
    directly returning the precomputed optimal solution.
    """
    return _OPTIMAL
