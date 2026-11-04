import numpy as np
from scipy.special import roots_laguerre

def construct():
    """
    Returns the 12 positive roots of the Laguerre polynomial L_12(x).
    These roots correspond to the standard orthogonal polynomial construction
    which is optimal for the uncertainty bound in this context.
    """
    N = 12
    roots, _ = roots_laguerre(N)
    return [float(r) for r in roots]
