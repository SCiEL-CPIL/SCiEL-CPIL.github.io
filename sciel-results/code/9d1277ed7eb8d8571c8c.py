"""Refined Cohn-Goncalves Laguerre-polynomial construction."""
import time
import numpy as np
from scipy.special import roots_genlaguerre

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a list of 12 positive double-root locations for the Laguerre combination.
    Uses the roots of the generalized Laguerre polynomial L_12^(1)(x).
    """
    start_time = time.time()
    deadline = 600.0
    safety_margin = 5.0
    
    m = 12
    alpha = 1.0
    
    try:
        # Compute roots of L_m^(alpha)(x)
        # roots_genlaguerre returns x (roots) and w (weights)
        x, w = roots_genlaguerre(m, alpha)
        
        # Ensure roots are positive and sorted
        # The roots of L_m^(alpha) for alpha > -1 are all positive and distinct.
        roots = sorted(x)
        
        # Verify constraints: 12 roots, all positive
        if len(roots) != 12:
            raise ValueError("Expected 12 roots")
        if any(r <= 0 for r in roots):
            raise ValueError("Roots must be positive")
            
        # Check deadline (unlikely to be needed, but required by directive)
        if time.time() + safety_margin > deadline:
            # Fallback to seed if we are somehow close to deadline (should not happen)
            return [4.0 * (i + 1) for i in range(12)]
            
        return [float(r) for r in roots]
        
    except Exception:
        # Fallback to seed program behavior if something goes wrong
        return [4.0 * (i + 1) for i in range(12)]
# EVOLVE-BLOCK-END
