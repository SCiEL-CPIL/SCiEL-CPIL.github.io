"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_genlaguerre

def construct():
    m = 12
    n = 1
    
    # Get the zeros of the generalized Laguerre polynomial L_m^(n)(x)
    x_zeros, _ = roots_genlaguerre(m, n)
    
    # Ensure they are positive and sorted
    x_zeros = np.sort(np.abs(x_zeros))
    
    return list(x_zeros)
# EVOLVE-BLOCK-END
