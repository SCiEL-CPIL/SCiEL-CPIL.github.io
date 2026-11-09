"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.special import roots_laguerre

# EVOLVE-BLOCK-START
def construct():
    """Return 12 positive double-root locations (zeros of L_12) for the
    Cohn-Goncalves Laguerre construction (m=12, n=1)."""
    # The optimal double-root locations are the zeros of the Laguerre polynomial L_12(x)
    # These are the nodes that minimize the Fourier sign uncertainty upper bound.
    roots = roots_laguerre(12)
    # roots_laguerre returns a tuple (x, w); x contains the 12 positive zeros
    return list(roots[0])
# EVOLVE-BLOCK-END
