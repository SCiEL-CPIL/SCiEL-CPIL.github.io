"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # Use Gauss-Legendre quadrature of order 8.
    # This provides high-order accuracy (exact for polynomials up to degree 15)
    # which is significantly better than the midpoint rule for smooth integrands.
    
    # Get nodes and weights for the interval [-1, 1]
    x, w = np.polynomial.legendre.leggauss(8)
    
    # Transform to the interval [0, 1]
    # t = (x + 1) / 2
    # dt = dx / 2
    # So weights are halved.
    nodes = (x + 1.0) / 2.0
    weights = w / 2.0
    
    # Ensure strict ordering and positivity constraints are met
    # leggauss returns nodes in increasing order and positive weights.
    # Sum of weights is 1.0.
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
