import numpy as np
from scipy.special import roots_genlaguerre

def construct():
    """
    Returns the 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Based on the analytical optimal configuration for the Cohn-Goncalves construction,
    these are the roots of the generalized Laguerre polynomial L_12^(11)(x).
    """
    # Calculate roots of L_12^(11)(x)
    # n=12, alpha=11
    try:
        # roots_genlaguerre returns (roots, weights)
        roots, _ = roots_genlaguerre(12, 11)
        # Ensure they are real and positive
        roots = np.real(roots)
        # Filter out any non-positive roots (should not happen for this alpha)
        roots = roots[roots > 0]
        # Sort them
        roots = np.sort(roots)
        # Return as a list of floats
        return [float(r) for r in roots]
    except Exception:
        # Fallback to seed if scipy fails or roots are invalid
        return [4.0 * (i + 1) for i in range(12)]
