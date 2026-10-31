import zlib

# EVOLVE-BLOCK-START
def priority(vector, n):
    # Use CRC32 for a fast, deterministic, well-distributed hash
    # This effectively randomizes the greedy order, which helps avoid
    # structural local optima in cap set construction.
    try:
        # bytes(vector) is fast for small tuples of small ints
        return float(zlib.crc32(bytes(vector)))
    except Exception:
        # Fallback to simple sum if bytes conversion fails (unlikely)
        return float(sum(vector))
# EVOLVE-BLOCK-END
