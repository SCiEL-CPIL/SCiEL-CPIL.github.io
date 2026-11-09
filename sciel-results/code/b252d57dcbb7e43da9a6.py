"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    The optimal solution for OneMax is a string of all '1's.
    Since the objective is deterministic and known (maximize count of '1's),
    we directly return the global optimum.
    """
    # Use NumPy vectorized approach to construct the string of '1's
    # This leverages C-level operations for efficiency
    arr = np.ones(N, dtype=np.uint8)
    # Convert to string: each element becomes '1'
    result = np.char.mod('1', arr)
    return ''.join(result)
# EVOLVE-BLOCK-END
