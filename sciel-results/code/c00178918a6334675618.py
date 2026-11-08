"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Returns the optimal step heights for the Erdos minimum overlap problem.
    
    The problem asks to minimize the maximum overlap of h with shifted 1-h.
    Let O_k be the overlap at shift k.
    For circular shifts, O_k = N/2 - S_k, where S_k is the auto-correlation.
    The average of S_k is N/4. Thus min S_k <= N/4.
    Consequently, max O_k >= N/2 - N/4 = N/4.
    The constant sequence h_i = 1/2 achieves S_k = N/4 for all k, 
    so max O_k = N/4.
    This is the global minimum.
    """
    return np.full(N, 0.5)
# EVOLVE-BLOCK-END
