import hashlib

def priority(vector, n):
    # Convert vector to bytes for hashing
    # Assuming vector is a tuple of ints 0, 1, 2
    b = bytes(vector)
    h = hashlib.sha256(b).digest()
    # Use first 8 bytes for a 64-bit integer
    # This provides a high-quality random permutation
    val = int.from_bytes(h[:8], 'little')
    return float(val)
