import zlib

def priority(vector, n):
    # Use a fast, deterministic hash to randomize the greedy order.
    # Lexicographic order (baseline) clusters similar vectors, causing
    # early conflicts and smaller final cap sets. Random ordering spreads
    # out the vectors, allowing the greedy algorithm to select a larger
    # set before hitting conflicts.
    key = str(vector).encode('utf-8')
    return float(zlib.crc32(key))
