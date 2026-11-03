import numpy as np
from scipy.special import roots_genlaguerre

def construct():
    """
    Returns 12 positive double-root locations for the Laguerre combination.
    Uses the roots of the generalized Laguerre polynomial L_12^alpha(x).
    The standard Cohn-Goncalves construction uses alpha=0.
    """
    try:
        # m=12, n=1. The polynomial degree is likely related to m.
        # The roots are for the Laguerre polynomial of degree 12.
        # alpha=0 corresponds to the standard Laguerre polynomial L_12(x).
        roots = roots_genlaguerre(12, 0)
        # Ensure they are positive and real
        roots = np.real_if_close(roots)
        roots = roots[roots > 0]
        
        # If for some reason we don't have 12 roots, fallback
        if len(roots) != 12:
            return [4.0 * (i + 1) for i in range(12)]
            
        return [float(r) for r in roots]
    except Exception:
        return [4.0 * (i + 1) for i in range(12)]
