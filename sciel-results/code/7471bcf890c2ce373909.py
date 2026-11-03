"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a sequence of N step heights in [0, 1] with sum N/2.
    Based on the analysis that the constant sequence h_i = 0.5 minimizes the 
    maximum overlap (equivalent to maximizing the minimum autocorrelation, 
    which is bounded by the average autocorrelation) for periodic domains,
    and also minimizes max autocorrelation.
    """
    return np.full(N, 0.5)
# EVOLVE-BLOCK-END
