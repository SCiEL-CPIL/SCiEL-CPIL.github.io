import numpy as np

def construct():
    # Get the roots of the Laguerre polynomial L_12(x)
    # laggauss(n) returns the nodes (roots) and weights for Laguerre-Gauss quadrature
    # The nodes are the roots of L_n(x)
    roots, _ = np.polynomial.laguerre.laggauss(12)
    
    # Ensure they are positive and sorted
    roots = sorted(roots)
    
    return roots
