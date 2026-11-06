"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import hashlib
import struct

# We use a deterministic hash to create a pseudo-random priority ordering.
# This avoids the systematic bias of "counting 1s" which may cluster vectors
# in subspaces that are mutually exclusive or mutually inclusive in bad ways.
# A random permutation often provides a more uniform exploration of the
# "conflict graph" of the cap set problem.

# We use a fixed seed/salt to ensure determinism across evaluations.
_SALT = b"cap_set_salt_v2"

def _vector_to_bytes(vector):
    """Convert a vector of {0,1,2} to a byte string for hashing."""
    # Pack the vector into a bytes object
    # Each element is 0, 1, or 2. We can pack 4 elements per byte to save space,
    # or just use a simple string representation. For n up to ~30, string is fine.
    # But for speed, we pack into a byte array.
    # Actually, just converting to a string and hashing is fast enough for n<50.
    return b"".join(chr(48 + x) for x in vector)

def priority(vector, n):
    # Create a deterministic pseudo-random priority.
    # We hash the vector content and the dimension n to ensure different n
    # get different (but stable) orderings.
    data = _vector_to_bytes(vector) + str(n).encode('ascii') + _SALT
    
    # Use SHA-256 for high quality mixing
    h = hashlib.sha256(data).digest()
    
    # Convert first 8 bytes to a 64-bit float for priority
    # We map the hash to [0.0, 1.0)
    # Using the first 53 bits for the mantissa of a double
    val = struct.unpack('<Q', h[:8])[0]
    
    # Return a float in [0, 1)
    # Higher is better, so we just return the normalized value.
    return (val % (2**53)) / float(2**53)

# EVOLVE-BLOCK-END
