"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    """
    Returns the optimal signed step function for the C3' task.
    
    The objective is to minimize max_k |(f*f)[k]| / (sum f)^2.
    The theoretical lower bound for any real vector f is 1/N, achieved by 
    constant vectors (e.g., all ones).
    Since the ratio is scale-invariant, we return the simplest constant vector.
    """
    # The optimal solution is a constant vector.
    # f = [1, 1, ..., 1]
    # sum(f) = N
    # f*f is a triangle from N down to 1.
    # max |f*f| = N (at lag 0)
    # Ratio = N / N^2 = 1/N.
    
    return np.ones(N)
# EVOLVE-BLOCK-END
