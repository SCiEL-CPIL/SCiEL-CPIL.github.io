"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_genlaguerre

def construct():
    # Use zeros of generalized Laguerre polynomial L_12^(1)(x)
    # n=1 suggests alpha=1, m=12
    roots, _ = roots_genlaguerre(12, 1)
    # Ensure positive and sort
    roots = sorted([float(r) for r in roots])
    return roots
# EVOLVE-BLOCK-END
