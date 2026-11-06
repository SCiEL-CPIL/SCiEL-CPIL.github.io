"""OneMax direct binary construction (100 bits).

The objective is to maximize the number of ones in a 100-bit binary string.
The optimal solution is a string of all ones. Since the task is deterministic
and the optimum is trivially known, we return the optimal string directly.
"""
import numpy as np
N = 100

# EVOLVE-BLOCK-START
def construct():
    """Return a 100-bit binary string with all ones (optimal for OneMax)."""
    # Use NumPy boolean array for vectorized construction
    bits = np.ones(N, dtype=bool)
    # Convert boolean array to string of '1's
    # np.char.mod or join can be used, but joining list of chars is standard
    # Since all are True, they map to '1'
    return "".join("1" if b else "0" for b in bits)
# EVOLVE-BLOCK-END
