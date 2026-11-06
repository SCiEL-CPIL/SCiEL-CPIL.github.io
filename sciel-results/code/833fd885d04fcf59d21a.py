# Pre-allocate a NumPy array of ones at module load time
_ones_array = np.ones(N, dtype=np.uint8)

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    Uses a pre-allocated NumPy array of ones, converting to string via byte decoding.
    """
    return _ones_array.tobytes().decode('ascii')
# EVOLVE-BLOCK-END
