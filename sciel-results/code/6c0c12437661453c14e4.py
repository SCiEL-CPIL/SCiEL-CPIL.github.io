"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_laguerre

def construct():
    """
    Construct the 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    
    The optimal root placement for the Cohn-Goncalves Laguerre-polynomial construction
    corresponds to the zeros of the Laguerre polynomial L_12(x). These roots minimize
    the associated uncertainty bound under the Laguerre weight e^{-x}.
    """
    # Compute the zeros of the Laguerre polynomial of degree 12
    # roots_laguerre(n) returns an array of zeros and an array of weights
    zeros, _ = roots_laguerre(12)
    
    # Return as a list of floats
    return [float(z) for z in zeros]
# EVOLVE-BLOCK-END
