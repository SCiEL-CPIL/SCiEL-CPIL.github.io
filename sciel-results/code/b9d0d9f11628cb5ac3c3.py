import numpy as np
import scipy.special

def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination.
    Uses the roots of the generalized Laguerre polynomial L_12^(1)(x) as the locations.
    """
    # roots_genlaguerre(n, alpha) returns the roots and weights of L_n^(alpha)(x)
    # We try alpha=1 as a refinement over the standard alpha=0
    roots, _ = scipy.special.roots_genlaguerre(12, 1)
    
    # The roots are already positive and sorted in increasing order.
    # Convert to a list of floats for the interface.
    return [float(r) for r in roots]
