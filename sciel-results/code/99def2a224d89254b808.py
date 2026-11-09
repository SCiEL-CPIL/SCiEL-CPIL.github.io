"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_genlaguerre

def construct():
    # The task suggests using the zeros of the Laguerre polynomial L_12^(-1/2)(x)
    # as the double-root locations z_i.
    # scipy.special.roots_genlaguerre(n, alpha) returns the roots of L_n^(alpha)(x).
    n = 12
    alpha = -0.5
    
    # Compute the roots
    x, w = roots_genlaguerre(n, alpha)
    
    # Convert to float and ensure they are sorted
    z = sorted([float(xi) for xi in x])
    
    return z
# EVOLVE-BLOCK-END
