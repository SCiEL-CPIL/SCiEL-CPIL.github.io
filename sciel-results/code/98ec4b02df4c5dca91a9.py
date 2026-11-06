"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import zlib

# EVOLVE-BLOCK-START
def priority(vector, n):
    """
    Prioritizes vectors using a pseudo-random hash order.
    This approach breaks the symmetry of the vector space, avoiding the
    clustering of vectors with similar structural properties (like low
    Hamming weight) that can lead to frequent arithmetic progression
    conflicts during greedy construction.
    """
    # Convert the tuple of small integers (0, 1, 2) to bytes
    data = bytes(vector)
    # Calculate a deterministic CRC32 checksum
    h = zlib.crc32(data)
    # Return negative hash to prioritize lower hash values
    return -float(h)
# EVOLVE-BLOCK-END
