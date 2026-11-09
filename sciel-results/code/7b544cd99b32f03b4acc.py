"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Constructs an 8-node Gauss-Legendre quadrature rule on the interval [0, 1].
    
    Strategy:
    The luminosity distance integral, after the substitution z' = z*x, becomes
    an integral over the fixed domain [0, 1]:
    D_L = (c/H0) * (1+z) * z * ∫_0^1 dx / E(z*x)
    
    Gauss-Legendre quadrature is spectrally accurate for smooth functions.
    For standard cosmological models (like CPL), the integrand 1/E(z*x) is smooth
    and analytic for z in the validation range. An 8-point Gauss-Legendre rule
    integrates polynomials up to degree 15 exactly and provides significantly
    higher accuracy than uniform midpoint rules for the same number of samples.
    
    We compute the standard Legendre nodes and weights for [-1, 1] and map them
    to [0, 1]. The weights are scaled to sum to 1.
    """
    n = 8
    # Get Gauss-Legendre nodes and weights for [-1, 1]
    # leggauss returns nodes in increasing order
    x_old, w_old = np.polynomial.legendre.leggauss(n)
    
    # Map nodes from [-1, 1] to [0, 1]
    # x_new = (x_old + 1) / 2
    nodes = (x_old + 1.0) / 2.0
    
    # Map weights from [-1, 1] (sum=2) to [0, 1] (sum=1)
    # w_new = w_old / 2
    weights = w_old / 2.0
    
    # Ensure nodes are strictly increasing (leggauss returns them sorted, but just in case)
    # and within [0, 1]
    # leggauss nodes are strictly inside (-1, 1), so mapped nodes are strictly inside (0, 1)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
