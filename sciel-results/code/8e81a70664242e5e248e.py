"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_genlaguerre

def construct():
    # The task specifies m=12 and n=1.
    # The theoretical optimal nodes for the Laguerre combination in this context
    # are the roots of the generalized Laguerre polynomial L_m^(alpha)(x).
    # For n=1 and the standard weight x*e^-x, alpha=1 is the appropriate parameter.
    # We compute the 12 roots of L_12^(1)(x).
    try:
        # roots_genlaguerre returns (x, w) where x are the roots
        roots, _ = roots_genlaguerre(12, 1)
        # Ensure the roots are sorted and returned as a list of floats
        return sorted(roots.tolist())
    except Exception:
        # Fallback to the seed value if scipy fails for any reason
        return [4.0 * (i + 1) for i in range(12)]
# EVOLVE-BLOCK-END
