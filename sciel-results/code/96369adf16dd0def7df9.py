import numpy as np
from scipy.special import roots_laguerre

def construct():
    """
    Returns the 12 positive double-root locations for the Laguerre combination.
    Based on the task description 'm=12' and '12 double roots', we use the roots
    of the Laguerre polynomial L_12(x), which are the natural critical points
    for this construction.
    """
    try:
        # Compute the roots of the Laguerre polynomial of degree 12
        roots, _ = roots_laguerre(12)
        
        # Ensure roots are positive (they should be for Laguerre) and sorted
        roots = np.sort(roots)
        
        # Convert to list of floats as required
        return [float(r) for r in roots]
    except Exception:
        # Fallback to seed-like values if scipy fails, though it shouldn't
        return [4.0 * (i + 1) for i in range(12)]
