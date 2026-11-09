"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from numpy.polynomial import laguerre

def construct():
    # Cohn-Goncalves Laguerre-polynomial construction:
    # The 12 double-root locations are the zeros of L_12(x)
    roots, _ = laguerre.laggauss(12)
    return list(roots)
# EVOLVE-BLOCK-END
