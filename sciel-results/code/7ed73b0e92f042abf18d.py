"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def construct():
    # Gauss-Legendre quadrature rule with 8 nodes on [0, 1].
    # This rule is exact for polynomials up to degree 15, providing
    # high accuracy for smooth integrands like those in cosmological
    # distance calculations, significantly outperforming the midpoint rule.
    
    # Standard Gauss-Legendre nodes on [-1, 1]
    x = np.array([
        -0.960289856497536,
        -0.796666477413626,
        -0.525532409916329,
        -0.183434642500288,
         0.183434642500288,
         0.525532409916329,
         0.796666477413626,
         0.960289856497536
    ])
    
    # Standard Gauss-Legendre weights on [-1, 1]
    w = np.array([
        0.101228536290376,
        0.222381034453374,
        0.313706645877887,
        0.362683783378362,
        0.362683783378362,
        0.313706645877887,
        0.222381034453374,
        0.101228536290376
    ])
    
    # Map nodes from [-1, 1] to [0, 1]
    nodes = 0.5 * (x + 1.0)
    
    # Scale weights by the length of the interval (0.5)
    weights = 0.5 * w
    
    # Ensure nodes are strictly increasing (they are by construction)
    # Ensure weights sum to 1 (they do by construction, up to floating point)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
