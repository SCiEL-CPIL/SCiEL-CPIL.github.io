"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.special import roots_laguerre
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Returns the 12 roots of the standard Laguerre polynomial L_12(x), which form the
    natural orthogonal-polynomial baseline for the Cohn-Goncalves construction.
    """
    deadline = time.time() + 570  # Safety margin before 630s wall-clock limit
    
    # The Cohn-Goncalves refined uncertainty inequality with m=12 double roots
    # and n=1 uses the Laguerre polynomial structure. The optimal double-root
    # locations that minimize the upper bound on C4 (determined by the last
    # sign change) are the zeros of the Laguerre polynomial L_12(x).
    #
    # L_12(x) = sum_{k=0}^{12} (-1)^k * binom(12, k) * x^k / k!
    # Its 12 positive roots are the natural nodes for Laguerre quadrature
    # and minimize the weighted L^2 norm, providing the physically consistent
    # baseline for this construction.
    
    try:
        x_roots, _ = roots_laguerre(12)
        result = [float(r) for r in x_roots]
    except Exception:
        # Fallback: use the arithmetic progression that scored 1.0
        result = [4.0 * (i + 1) for i in range(12)]
    
    # Verify: 12 positive, strictly increasing roots
    if len(result) != 12 or any(r <= 0 for r in result) or any(result[i] >= result[i+1] for i in range(11)):
        result = [4.0 * (i + 1) for i in range(12)]
    
    # Deadline guard (should never trigger for this fast computation)
    if time.time() > deadline:
        return result
    
    return result
# EVOLVE-BLOCK-END
