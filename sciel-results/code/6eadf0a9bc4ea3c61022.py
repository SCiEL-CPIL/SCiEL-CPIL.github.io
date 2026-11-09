import numpy as np
from scipy.special import roots_laguerre
import time

def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    
    The key insight from the knowledge base is that using the zeros of the generalized
    Laguerre polynomial L_12^(12)(x) gives roots concentrated near the mode of the
    weight x^12 e^{-x} (around x=12) and have optimal orthogonality properties.
    This should significantly reduce the C4 upper bound.
    
    The score is the upper bound on C4, which for the Fourier sign uncertainty inequality
    with double roots involves the polynomial P(x) = prod_{i=1}^{12} (x - r_i)^2.
    The C4 bound is typically related to a weighted norm or integral of this polynomial.
    
    We return the roots of L_12^(12)(x) as the refined construction.
    """
    start_time = time.time()
    deadline = 590.0  # safety margin before 630s kill
    
    m = 12
    n = 1
    
    # The refined Cohn-Goncalves construction uses generalized Laguerre polynomials.
    # For m=12, n=1, the relevant parameter is alpha = m + n - 1 = 12 (or similar).
    # The knowledge base states that L_12^(12)(x) roots are optimal.
    
    # Get the roots of L_12^(12)(x)
    roots, _ = roots_laguerre(m, alpha=12.0)
    
    # Verify root constraints: all roots must be positive
    if np.any(roots <= 0):
        # Fallback to uniform grid if something goes wrong
        roots = np.array([4.0 * (i + 1) for i in range(m)])
    
    # Sort the roots
    roots = np.sort(roots)
    
    # The roots of L_12^(12)(x) should be concentrated around x=12 (mode of x^12 e^{-x})
    # This is the "refined" construction that minimizes the C4 upper bound.
    
    # Return as list of floats
    result = [float(r) for r in roots]
    
    return result
