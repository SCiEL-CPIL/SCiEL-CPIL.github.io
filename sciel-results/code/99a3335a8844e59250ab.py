"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from numpy.polynomial.legendre import leggauss


def construct():
    # 8-point Gauss-Legendre quadrature on [0,1]
    # Map from [-1,1] to [0,1]: t = (x+1)/2, dt = dx/2
    x, w = leggauss(8)
    nodes = (x + 1.0) / 2.0
    weights = w / 2.0
    # Ensure strictly increasing (leggauss returns in increasing order on [-1,1])
    # Nodes are in (0,1) since leggauss nodes are in (-1,1)
    # Weights are positive and sum to 1
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
