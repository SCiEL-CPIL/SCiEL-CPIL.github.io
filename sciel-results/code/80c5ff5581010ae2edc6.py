"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.special import roots_genlaguerre

# EVOLVE-BLOCK-START
def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Uses zeros of the Laguerre polynomial L_12(x) as the root locations.
    The score (upper bound on C4) is computed as 1/sqrt(prod(roots))."""
    
    try:
        # Get zeros of the standard Laguerre polynomial L_12(x)
        x, w = roots_genlaguerre(12, 0)
        x = sorted(x)
        return list(x)
    except Exception:
        # Fallback to seed
        return [4.0 * (i + 1) for i in range(12)]
# EVOLVE-BLOCK-END
