import numpy as np
from scipy.special import roots_genlaguerre

def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Returns a list of 12 positive root locations.
    """
    # The optimal roots for the Laguerre-based uncertainty bound are the zeros
    # of the generalized Laguerre polynomial L_12^(1)(x).
    # This corresponds to the solution of the variational problem for the
    # associated quadratic form.
    
    # n=12, alpha=1
    roots, _ = roots_genlaguerre(12, 1)
    
    # Ensure they are returned as a list of floats
    return [float(r) for r in roots]
