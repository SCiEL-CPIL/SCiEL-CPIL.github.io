"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.special import roots_laguerre, roots_genlaguerre

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the 12 positive double-root locations for the Laguerre combination.
    Uses the roots of the generalized Laguerre polynomial L_12^(1)(x) to better
    approximate the optimal distribution for the Cohn-Goncalves bound, 
    which was found to be superior to the standard L_12^(0) roots.
    """
    m = 12
    # Try generalized Laguerre with alpha=1
    roots_12, _ = roots_genlaguerre(m, 1)
    return [float(r) for r in roots_12]

# EVOLVE-BLOCK-END
