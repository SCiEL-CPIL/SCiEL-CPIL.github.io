"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import roots_genlaguerre

_START = time.time()
_DEADLINE = 590.0  # safety margin before 630s kill

def _deadline_reached():
    return (time.time() - _START) >= _DEADLINE

def construct():
    """
    Return 12 positive double-root locations for the Laguerre combination
    (m=12, n=1) that minimize the C4 upper bound on Fourier sign uncertainty.
    
    Strategy: Use the zeros of the generalized Laguerre polynomial L_12^{(1)}(x),
    which are the natural optimal nodes for this construction.
    """
    # Get zeros of L_12^{(1)}(x)
    # roots_genlaguerre(n, alpha) returns zeros of L_n^{(alpha)}(x)
    try:
        roots, _ = roots_genlaguerre(12, 1.0)
        roots = np.sort(roots)
        # Ensure all positive (they should be for Laguerre)
        roots = roots[roots > 0]
        if len(roots) == 12:
            return [float(r) for r in roots]
    except Exception:
        pass
    
    # Fallback: arithmetic progression (seed behavior)
    return [4.0 * (i + 1) for i in range(12)]
# EVOLVE-BLOCK-END
