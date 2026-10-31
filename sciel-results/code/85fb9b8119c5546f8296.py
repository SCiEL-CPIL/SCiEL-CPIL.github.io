import hashlib

def priority(vector, n):
    # Convert vector to bytes for hashing
    # vector is a tuple of 0, 1, 2
    b = bytes(vector)
    # Use blake2b for fast, high-quality 64-bit hash
    h = hashlib.blake2b(b, digest_size=8)
    # Convert to int
    return int.from_bytes(h.digest(), byteorder='big')
