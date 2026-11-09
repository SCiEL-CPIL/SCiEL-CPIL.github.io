"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_genlaguerre

def construct():
    # Use the exact zeros of the generalized Laguerre polynomial L_12^(1)(x)
    # These are 12 positive real roots that leverage orthogonality properties
    # of classical orthogonal polynomials, which are known to be optimal
    # for uncertainty minimization problems.
    roots = roots_genlaguerre(12, 1)
    # roots_genlaguerre returns (eigenvalues, eigenfunctions)
    # The eigenvalues are the zeros of L_12^(1)(x)
    root_locations = list(roots[0].real)
    # Ensure they are sorted and positive
    root_locations = sorted([float(r) for r in root_locations if r > 0])
    # We need exactly 12 roots
    if len(root_locations) == 12:
        return root_locations
    # Fallback: if something went wrong, return the seed
    return [4.0 * (i + 1) for i in range(12)]
# EVOLVE-BLOCK-END
