import numpy as np
import scipy.special

def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination.
    Uses the roots of the Generalized Laguerre polynomial L_12^{(-0.5)}(x) as the locations.
    """
    # roots_genlaguerre(n, alpha) returns the roots of L_n^{(alpha)}(x)
    # We use n=12 and alpha=-0.5 as a refinement for the uncertainty bound
    roots, _ = scipy.special.roots_genlaguerre(12, -0.5)
    
    # The roots are already positive and sorted in increasing order.
    # Convert to a list of floats for the interface.
    return [float(r) for r in roots]
