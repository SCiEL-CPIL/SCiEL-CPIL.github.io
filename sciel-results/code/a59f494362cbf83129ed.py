import numpy as np
from scipy.special import roots_genlaguerre

def construct():
    """
    Returns 12 double-root locations z_1..z_12.
    Strategy: Use the positive zeros of the generalized Laguerre polynomial L_12^(-1/2)(x).
    These are the optimal nodes for Laguerre quadrature/interpolation of degree 12,
    which likely minimizes the residual polynomial q(x) in the context of the
    Laguerre expansion basis used for g(x).
    """
    # n=12 corresponds to 12 roots. alpha=-0.5 matches the problem statement.
    roots = roots_genlaguerre(12, -0.5)
    
    # Ensure they are floats and sorted
    z = sorted([float(r) for r in roots])
    
    # Basic validity check: finite, > 0, distinct
    if len(z) != 12:
        # Fallback to seed if something goes wrong
        return [4.0 * (i + 1) for i in range(12)]
        
    if any(not np.isfinite(zi) or zi <= 0 for zi in z):
        return [4.0 * (i + 1) for i in range(12)]
        
    if len(set(z)) != 12:
        return [4.0 * (i + 1) for i in range(12)]
        
    return z
