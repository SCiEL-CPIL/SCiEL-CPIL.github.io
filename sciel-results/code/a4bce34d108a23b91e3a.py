import numpy as np
import scipy.special

def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination.
    Uses the roots of the generalized Laguerre polynomial L_12^(alpha)(x) as the locations.
    """
    # Use generalized Laguerre polynomial with alpha=1.0
    # This often appears in refined uncertainty bounds involving derivatives or weighted norms.
    roots, _ = scipy.special.roots_genlaguerre(12, 1.0)
    
    # The roots are already positive and sorted in increasing order.
    # Convert to a list of floats for the interface.
    return [float(r) for r in roots]
