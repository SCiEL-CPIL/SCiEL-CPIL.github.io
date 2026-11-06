"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import hashlib
import struct

def priority(vector, n):
    """
    Assign a deterministic, pseudo-random priority to each vector in {0,1,2}^n
    using a cryptographic hash of the vector encoded as a byte string.
    
    The idea: use a hash-based permutation of the hypercube to create a
    globally unique, deterministic ordering that avoids local clustering.
    This creates a uniform exploration order that may find larger cap sets
    than simple lexicographic or balance-based orderings.
    
    We hash the vector (as a tuple of integers) to get a 64-bit integer,
    then map it to a float in [0, 1). Higher values = higher priority.
    """
    # Encode the vector into a byte string for hashing
    # Use a simple encoding: each element as one byte
    vec_bytes = bytes(vector)
    
    # Use SHA-256 for a strong hash, take first 8 bytes as a 64-bit integer
    h = hashlib.sha256(vec_bytes).digest()
    # Convert first 8 bytes to an unsigned 64-bit integer
    val = struct.unpack('>Q', h[:8])[0]
    
    # Map to [0, 1)
    # 2^64 is the max value, so divide by 2^64
    return val / (1 << 64)
# EVOLVE-BLOCK-END
