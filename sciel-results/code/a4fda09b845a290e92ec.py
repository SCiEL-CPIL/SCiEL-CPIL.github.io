"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_legendre


def construct():
    # Gauss-Legendre quadrature with 8 nodes on [0,1]
    # This is optimal for smooth integrands and works well for cosmological E(z)
    x, w = roots_legendre(8)
    # Transform from [-1,1] to [0,1]: t = (x+1)/2, dt = dx/2
    nodes = (x + 1.0) / 2.0
    weights = w / 2.0
    # Ensure nodes are strictly increasing and in [0,1]
    nodes = np.sort(nodes)
    # Ensure weights sum to 1
    weights = weights / np.sum(weights)
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
