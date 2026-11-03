import numpy as np
from numpy.polynomial import laguerre

def construct():
    # Compute the roots of the Laguerre polynomial L_12(x)
    # laggauss returns points and weights for integration, but we need roots of the polynomial
    # We can get the roots using the eigenvalues of the companion matrix or numpy.roots
    # However, numpy.polynomial.laguerre.laggauss gives the Gauss-Laguerre quadrature points,
    # which are the roots of L_m(x) for the standard weight e^-x.
    
    # For m=12, the roots of L_12(x) are the nodes of the Gauss-Laguerre quadrature.
    x, w = laguerre.laggauss(12)
    
    # The roots are positive.
    # Ensure they are sorted (laggauss returns them sorted)
    roots = list(x)
    
    return roots
