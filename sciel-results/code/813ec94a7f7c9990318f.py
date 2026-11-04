import numpy as np
from scipy.special import roots_laguerre

def construct():
    """
    Returns the roots of the Laguerre polynomial L_12, which are the
    optimal locations for the double roots in the Cohn-Goncalves
    construction, minimizing the C4 uncertainty bound.
    """
    roots, _ = roots_laguerre(12)
    return [float(r) for r in roots]
