"""Refined Cohn-Goncalves Laguerre construction for Fourier sign uncertainty."""
import numpy as np
from scipy.special import roots_laguerre
import time

def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1)."""
    deadline = time.time() + 600.0
    
    # Primary approach: zeros of the Laguerre polynomial L_12(x)
    # In the Cohn-Goncalves construction for Fourier uncertainty,
    # the optimal roots are typically related to Laguerre polynomial zeros
    try:
        r, _ = roots_laguerre(12)
        root_list = sorted([float(x) for x in r])
        if len(root_list) == 12 and all(x > 0 for x in root_list):
            return root_list
    except Exception:
        pass
    
    # Secondary: try generalized Laguerre or other natural root sets
    # Zeros of L_12^(0)(x) are the standard ones; try L_12^(alpha) for small alpha
    for alpha in [0.5, 1.0, 2.0]:
        try:
            # Use numpy's roots of Laguerre via companion matrix
            # Build the monic Laguerre polynomial L_12^alpha
            from scipy.special import eval_genlaguerre
            # Get coefficients
            n = 12
            alpha_val = alpha
            # Build using recurrence or direct formula
            # Simpler: use scipy's roots_genlaguerre if available
            from scipy.special import roots_genlaguerre
            r, _ = roots_genlaguerre(n, alpha_val)
            root_list = sorted([float(x) for x in r])
            if len(root_list) == 12 and all(x > 0 for x in root_list):
                return root_list
        except Exception:
            continue
    
    # Fallback to seed
    return [4.0 * (i + 1) for i in range(12)]
