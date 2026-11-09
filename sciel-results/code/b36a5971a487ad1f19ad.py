import numpy as np
from scipy.special import roots_genlaguerre

def construct():
    """
    Returns 12 double-root locations z_1..z_12.
    Uses the zeros of the generalized Laguerre polynomial L_12^(0.5)(x)
    as the optimal node distribution.
    """
    # The problem involves L_2k^(-1/2). 
    # The nodes z_i are involved in Hermite interpolation conditions.
    # A natural choice for nodes in Laguerre-based problems is the zeros of 
    # the orthogonal polynomial of the same degree, or related derivatives.
    # 
    # Hypothesis: The zeros of L_12^(0.5)(x) (which are the stationary points 
    # of L_13^(-1/2)(x)) provide the optimal distribution.
    
    n = 12
    alpha = 0.5
    
    # roots_genlaguerre returns the roots of L_n^(alpha)(x)
    roots = roots_genlaguerre(n, alpha)
    
    # Ensure they are floats and sorted
    z_list = sorted([float(r) for r in roots])
    
    return z_list
