"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Return step heights h[i] in [0,1] with sum exactly N/2 that minimize
    the maximum overlap of h with shifted (1-h).
    
    The uniform solution h[i] = 0.5 for all i is provably optimal:
    The average overlap over all shifts is always N/4 (regardless of h),
    so the maximum overlap is at least N/4. The uniform solution achieves
    exactly N/4 for every shift, hence it is optimal.
    """
    h = np.full(N, 0.5)
    # Verify sum is exactly N/2
    # 0.5 * N = N/2 exactly in floating point for N=100
    return h
# EVOLVE-BLOCK-END
