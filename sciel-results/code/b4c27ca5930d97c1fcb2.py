"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    h = np.full(N, 1.0, dtype=np.float64)
    return h
# EVOLVE-BLOCK-END
