"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_genlaguerre
import time

def construct():
    """
    Return 12 positive double-root locations for the Laguerre combination
    (m=12, n=1) that minimize the C4 upper bound.
    
    Uses the zeros of the generalized Laguerre polynomial L_12^(alpha)(x)
    with large alpha. Large alpha spreads the roots over a wider range,
    increasing the product of roots and thus decreasing the 1/sqrt(product)
    bound. We search over alpha to find the configuration that maximizes
    the product of roots (minimizes the bound).
    """
    start_time = time.time()
    deadline = 600.0
    safety_margin = 30.0  # seconds before deadline to stop
    
    m = 12  # number of double roots
    
    best_product = 0.0
    best_roots = None
    
    # Search over alpha values - larger alpha spreads roots more
    # Start with a range of alpha values and refine
    alpha_candidates = [12, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
    
    for alpha in alpha_candidates:
        if time.time() > deadline - safety_margin:
            break
        
        try:
            x, w = roots_genlaguerre(m, alpha)
            
            # Verify all roots are positive
            if np.any(x <= 0):
                continue
            
            # Compute the product of roots
            # Use log to avoid overflow
            log_product = np.sum(np.log(x))
            
            if np.exp(log_product) > best_product:
                best_product = np.exp(log_product)
                best_roots = [float(xi) for xi in x]
        except Exception:
            continue
    
    # If no valid roots found, fall back to standard Laguerre
    if best_roots is None:
        x, w = roots_genlaguerre(m, m)
        best_roots = [float(xi) for xi in x]
    
    # Ensure roots are sorted in ascending order
    best_roots.sort()
    
    return best_roots
# EVOLVE-BLOCK-END
