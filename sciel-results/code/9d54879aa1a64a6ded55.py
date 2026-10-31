import zlib

def priority(vector, n):
    # Create a deterministic hash of the vector
    # Convert vector to an integer in base 3
    val = 0
    for x in vector:
        val = val * 3 + x
    # Use CRC32 to scramble the order
    # We want higher priority to be added first.
    # So we return the hash value.
    h = zlib.crc32(str(val).encode())
    return float(h)
