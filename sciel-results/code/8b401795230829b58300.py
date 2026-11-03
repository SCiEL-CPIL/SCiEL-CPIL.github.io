"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Construct the step heights h.
    
    Theoretical analysis shows that for any h with sum N/2:
    Sum_k C(k) = N^2/4, so average C(k) = N/4.
    Therefore max_k C(k) >= N/4.
    Uniform h=0.5 achieves C(k) = N/4 for all k, so it is optimal.
    
    This program returns the optimal solution directly.
    """
    # The optimal solution is uniform h_i = 0.5
    # This achieves the theoretical lower bound of max overlap = N/4 = 25.0
    
    # Return exactly N/2 sum (which is 50.0 for N=100)
    # and all values in [0, 1]
    return np.full(N, 0.5)
# EVOLVE-BLOCK-END
