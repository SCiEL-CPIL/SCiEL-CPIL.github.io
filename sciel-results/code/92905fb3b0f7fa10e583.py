import numpy as np
from scipy.special import roots_genlaguerre

def construct():
    """
    Returns the 12 positive roots of the generalized Laguerre polynomial L_12^(1)(x).
    These are the optimal nodes for the weight function x * exp(-x).
    """
    # n=12, alpha=1
    # roots_genlaguerre returns (roots, weights)
    roots, _ = roots_genlaguerre(12, 1)
    # Ensure they are positive and sorted
    roots = np.sort(roots)
    return roots.tolist()
