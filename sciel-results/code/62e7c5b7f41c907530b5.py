import numpy as np
from scipy.special import roots_genlaguerre

def construct():
    # Use the exact zeros of the generalized Laguerre polynomial L_12^(1)(x)
    # as the root locations. The parameter alpha=1 corresponds to the weight
    # x^1 * exp(-x) on [0, infinity), which is the appropriate choice for the
    # Cohn-Goncalves Laguerre-polynomial construction in the 1D Fourier sign
    # uncertainty problem with m=12 double roots.
    
    m = 12
    alpha = 1.0
    
    # Compute the roots of the generalized Laguerre polynomial L_m^(alpha)(x)
    # using SciPy's Gauss-Laguerre quadrature routine, which returns the nodes
    # (roots) and weights of the corresponding quadrature rule.
    roots, weights = roots_genlaguerre(m, alpha)
    
    # Ensure roots are positive and sorted (they should be by construction)
    roots = np.sort(roots)
    
    # Return as a list of Python floats
    return [float(r) for r in roots]
