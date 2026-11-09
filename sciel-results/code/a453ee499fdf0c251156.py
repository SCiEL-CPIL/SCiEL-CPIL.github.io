"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np
from scipy.special import roots_genlaguerre

def construct():
    """
    Return 12 double-root locations z_1..z_12.
    Strategy: Use the 12 smallest stationary points of the generalized Laguerre polynomial L_24^(-1/2)(x).
    The stationary points are the roots of the derivative L_24^(-1/2)'(x) = -L_23^(1/2)(x).
    We select the 12 smallest positive roots of L_23^(1/2)(x).
    This leverages the orthogonality structure and interlacing properties of Laguerre polynomials.
    """
    # Deadline guard (though this function is very fast)
    # The heavy lifting is in the evaluator, not here.
    
    # We need roots of L_23^(1/2)(x)
    n = 23
    alpha = 0.5
    
    try:
        # roots_genlaguerre returns (roots, weights)
        roots, _ = roots_genlaguerre(n, alpha)
        
        # Extract real parts (they should be real)
        real_roots = [r.real for r in roots]
        
        # Sort to get the smallest 12
        real_roots.sort()
        
        # Take the first 12
        z = [float(x) for x in real_roots[:12]]
        
        # Ensure they are positive and distinct
        # roots_genlaguerre guarantees distinct positive roots for alpha > -1
        return z
        
    except Exception:
        # Fallback to integers if scipy fails or produces issues
        return [float(i) for i in range(1, 13)]
# EVOLVE-BLOCK-END
