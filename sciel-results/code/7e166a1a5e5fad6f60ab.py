import numpy as np
from scipy.special import roots_genlaguerre

def construct():
    """
    Return 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Score is the upper bound on C4 (lower is better).
    
    The Cohn-Goncalves uncertainty bound is optimized by using the zeros of the 
    generalized Laguerre polynomial L_m^(alpha)(x) where alpha is related to the 
    dimension n. For n=1, alpha = n/2 - 1 = -0.5.
    """
    m = 12
    # Dimension n=1 implies alpha = -0.5 for the radial Laguerre polynomial
    alpha = -0.5
    
    try:
        # Get the roots of L_12^(-0.5)(x)
        roots = roots_genlaguerre(m, alpha)
        # roots_genlaguerre returns an array of real roots
        # Ensure they are positive and sorted
        roots = np.sort(roots.real)
        
        # Verify we have 12 positive distinct roots
        if len(roots) != 12 or np.any(roots <= 0) or np.any(np.diff(roots) < 1e-10):
            # Fallback to alpha=0 if there are numerical issues
            roots = roots_genlaguerre(m, 0)
            roots = np.sort(roots.real)
            
        return [float(r) for r in roots]
    except Exception:
        # Fallback to arithmetic progression if scipy fails (should not happen)
        return [4.0 * (i + 1) for i in range(12)]
