"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import hashlib

def priority(vector, n):
    """
    Assign a deterministic, pseudo-random priority to each vector in {0,1,2}^n
    using a fast cryptographic hash (BLAKE2b) of the vector.
    
    BLAKE2b is faster than SHA-256 and provides a high-quality permutation
    of the input space. This ensures a uniform, non-clustering order for
    the greedy cap set construction.
    """
    # Convert vector to bytes. Elements are 0, 1, 2, so they fit in a byte.
    vec_bytes = bytes(vector)
    
    # Use BLAKE2b with a digest size of 8 bytes (64 bits) for speed.
    # This is a strong, fast hash suitable for this purpose.
    h = hashlib.blake2b(vec_bytes, digest_size=8)
    
    # Convert the digest to an integer
    val = int.from_bytes(h.digest(), 'big')
    
    # Map to [0, 1)
    return val / (1 << 64)
# EVOLVE-BLOCK-END
