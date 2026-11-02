"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np

N = 100

def construct():
    """
    Constructs the optimal sequence of step heights.
    
    The problem asks to minimize the maximum overlap of h with shifted (1-h).
    For the circular overlap metric O_k = sum_i h_i (1-h_{i+k}), the average
    overlap over all non-zero shifts k is exactly N/4 (given sum(h) = N/2).
    Since the maximum overlap must be at least the average overlap, the lower
    bound is N/4.
    
    The constant sequence h_i = 0.5 achieves an overlap of exactly N/4 for all
    shifts k. Therefore, it is the global optimum.
    
    We return this constant sequence directly.
    """
    return np.full(N, 0.5)

# EVOLVE-BLOCK-END
