import zlib

def priority(vector, n):
    """
    Assigns a priority to a vector based on a stable hash of its contents.
    This creates a pseudo-random ordering for the greedy selection process,
    which is known to yield larger cap sets than lexicographic ordering.
    """
    # Convert the tuple of integers (0, 1, 2) to a bytes object for fast hashing
    # bytes() requires integers in range 0-255, which 0,1,2 satisfy.
    try:
        b = bytes(vector)
    except (TypeError, ValueError):
        # Fallback for unexpected input types
        b = str(vector).encode('utf-8')
    
    # zlib.crc32 returns an unsigned integer
    # Convert to float for the interface
    return float(zlib.crc32(b))
