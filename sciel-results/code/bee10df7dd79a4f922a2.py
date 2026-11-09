import math
import time
import numpy as np

def construct():
    # Use the 12 roots of the Laguerre polynomial L_12^(-1/2)(x).
    # These nodes are clustered near the origin (max ~ 40-50),
    # which is much smaller than the Bessel nodes (max ~ 1400).
    # This tighter clustering may reduce the largest root B of the quotient q(x).
    deadline = time.time() + 600.0
    
    # numpy.polynomial.laguerre.laggauss(n) returns the nodes (roots) and weights
    # for the Gauss-Laguerre quadrature, which corresponds to L_n^(-1/2)(x).
    nodes, weights = np.polynomial.laguerre.laggauss(12)
    
    # Ensure they are floats and sorted
    z = sorted([float(zi) for zi in nodes])
    
    # Safety: verify constraints
    if any(not (math.isfinite(zi) and zi > 0) for zi in z):
        raise ValueError("Invalid nodes")
    if len(set(z)) != 12:
        raise ValueError("Nodes not pairwise distinct")
        
    return z
