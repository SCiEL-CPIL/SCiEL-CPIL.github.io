import numpy as np
from scipy.special import roots_laguerre

def construct():
    """
    Constructs 12 positive double-root locations for the Laguerre combination.
    Uses the roots of the generalized Laguerre polynomial L_12^(23)(x),
    which corresponds to the refined Cohn-Goncalves construction for 
    Fourier sign uncertainty with high alpha to handle boundary effects.
    """
    n = 12
    alpha = 2 * n - 1  # 23
    
    # Get roots of L_n^(alpha)(x)
    # roots_laguerre(n, alpha) returns roots of L_n^(alpha)(x)
    roots = roots_laguerre(n, alpha)
    
    # Ensure they are sorted and positive
    roots = np.sort(roots)
    
    # Return as a list of floats
    return [float(r) for r in roots]
