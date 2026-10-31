"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    """Return step heights in [0,1] with sum exactly N/2, minimizing max overlap."""
    # The optimal solution for minimizing max_{i,j} h[i]*(1-h[j]) 
    # subject to sum(h)=N/2 and 0<=h<=1 is h[i]=0.5 for all i.
    # This gives score = 0.5 * 0.5 = 0.25.
    return np.full(N, 0.5)
# EVOLVE-BLOCK-END
