"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    # Alternating signed step function to minimize autoconvolution magnitude
    heights = np.ones(N)
    heights[1::2] = -1.0
    return heights
# EVOLVE-BLOCK-END
