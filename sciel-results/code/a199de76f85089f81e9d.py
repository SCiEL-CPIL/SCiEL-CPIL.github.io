"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_genlaguerre
import time

def construct():
    """
    Return 12 positive double-root locations for the Laguerre combination
    (m=12, n=1) that minimize the C4 upper bound.
    
    Uses the zeros of the generalized Laguerre polynomial L_12^(12)(x),
    which are the Gaussian quadrature nodes for weight x^12 * exp(-x).
    These nodes are optimal for moment matching under the Laguerre inner product.
    """
    m = 12
    alpha = m  # alpha = m for the weight x^m * exp(-x)
    
    # Get the roots of L_m^(alpha)(x)
    # roots_genlaguerre(n, alpha) returns the n roots of L_n^(alpha)(x)
    x, w = roots_genlaguerre(m, alpha)
    
    # The roots are already positive and sorted in ascending order
    # Ensure they are returned as a list of floats
    roots = [float(xi) for xi in x]
    
    return roots
# EVOLVE-BLOCK-END
