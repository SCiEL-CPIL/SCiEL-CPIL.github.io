import numpy as np
import scipy.special

def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination.
    Uses the roots of the Laguerre polynomial L_12(x) as the locations.
    """
    # Use generalized Laguerre roots with alpha=1.
    # This corresponds to the weight x^1 * e^(-x), which often appears
    # in refined uncertainty bounds involving derivatives or higher moments.
    roots, _ = scipy.special.roots_genlaguerre(12, 1)
    
    # The roots are already positive and sorted in increasing order.
    # Convert to a list of floats for the interface.
    return [float(r) for r in roots]
