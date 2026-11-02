"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Returns the optimal sequence of step heights.
    
    The problem is to minimize the maximum overlap O_k = sum h_i (1 - h_{i+k})
    subject to h_i in [0, 1] and sum h_i = N/2.
    
    The sum of overlaps over all shifts k=0..N-1 is:
    Sum_k O_k = Sum_k Sum_i h_i (1 - h_{i+k})
             = Sum_i h_i Sum_k (1 - h_{i+k})
             = Sum_i h_i (N - N/2)
             = (N/2) * (N/2)
             = N^2 / 4
    
    The average overlap is (N^2 / 4) / N = N / 4.
    Therefore, the maximum overlap must be at least N / 4.
    
    The constant sequence h_i = 0.5 yields:
    O_k = Sum_i 0.5 * (1 - 0.5) = Sum_i 0.25 = N * 0.25 = N / 4.
    
    Since the maximum overlap is exactly equal to the average overlap (N/4)
    for all k, this sequence achieves the theoretical minimum.
    No other sequence can have a maximum overlap lower than N/4.
    """
    return np.full(N, 0.5)

# EVOLVE-BLOCK-END
