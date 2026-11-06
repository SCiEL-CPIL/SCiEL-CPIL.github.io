"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 100

# Pre-allocate a NumPy array containing the ASCII code for '1' at module load time
_ones_bytes = np.full(N, 49, dtype=np.uint8)

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    Uses a pre-allocated NumPy array of ASCII '1' bytes, converting to string only upon invocation.
    """
    # Convert the pre-allocated byte array to a string via a single C-level call
    return _ones_bytes.tobytes().decode('ascii')
# EVOLVE-BLOCK-END
