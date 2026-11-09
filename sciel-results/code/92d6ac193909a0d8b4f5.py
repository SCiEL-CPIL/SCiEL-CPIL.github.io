"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 100

# EVOLVE-BLOCK-START
def construct():
    # Represent the 100-bit vector as a NumPy array of uint8 initialized to 1
    arr = np.ones(N, dtype=np.uint8)
    # Cast it to a string using np.char.mod
    return np.char.mod('1', arr).join('')
# EVOLVE-BLOCK-END
