"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Returns the uniform step height distribution h_i = 0.5 for all i.
    
    Theoretical Analysis:
    The overlap for shift k is C(k) = sum_i h_i (1 - h_{i+k}).
    The sum of overlaps over all k is sum_k C(k) = N^2/4.
    The average overlap is N/4.
    The maximum overlap is bounded below by the average overlap.
    The uniform distribution h_i = 0.5 yields C(k) = N/4 for all k.
    Thus, the uniform distribution is the unique global minimizer of the maximum overlap.
    Returning the exact uniform solution ensures the raw value is exactly N/4,
    maximizing the score (seed/value).
    """
    return np.full(N, 0.5)

# EVOLVE-BLOCK-END
