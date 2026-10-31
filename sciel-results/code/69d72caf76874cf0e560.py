"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    """Construct optimal signed step function minimizing max|autoconv|/integral^2.
    
    For any real vector x, by Cauchy-Schwarz: (sum x)^2 <= N * sum(x^2) = N * autoconv[0].
    Thus max|autoconv|/integral^2 >= autoconv[0]/integral^2 >= 1/N.
    Equality holds iff all x[i] are equal. So the optimal is a constant vector.
    """
    # The optimal solution is a constant vector (all ones or all -1s).
    # Using all ones gives integral = N, autoconv[0] = N, C3' = N/N^2 = 1/N.
    return np.ones(N)
# EVOLVE-BLOCK-END
