"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_laguerre
import time

def construct():
    """
    Return 12 positive double-root locations that minimize the upper bound
    on C4 for the Cohn-Goncalves Laguerre polynomial construction (m=12, n=1).
    
    Uses the zeros of the Laguerre polynomial L_12 as the root locations,
    which are the natural choice for this construction and should minimize
    the uncertainty bound.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin before 630s kill
    
    try:
        # Get the 12 zeros of the Laguerre polynomial L_12
        # These are positive and are the natural root locations
        # for the Cohn-Goncalves Laguerre construction
        x, w = roots_laguerre(12)
        roots = np.sort(x).tolist()
        
        # Verify constraints: all roots must be positive
        if any(r <= 0 for r in roots):
            raise ValueError("Roots must be positive")
        
        # Verify we have exactly 12 roots
        if len(roots) != 12:
            raise ValueError("Must have exactly 12 roots")
        
        return roots
    
    except Exception:
        # Fallback: use a reasonable set of positive roots
        # Based on the pattern of Laguerre zeros but hardcoded
        # for safety
        return [0.341, 1.248, 2.716, 4.756, 7.376, 10.582,
                14.379, 18.770, 23.756, 29.333, 35.501, 42.254]
# EVOLVE-BLOCK-END
