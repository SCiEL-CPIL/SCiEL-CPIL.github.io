import zlib

def priority(vector, n):
    # Prioritize vectors with more zeros (sparsity)
    # This aims to structurally define the space early,
    # potentially reducing accidental line formations later.
    zeros = vector.count(0)
    
    # Use hash for tie-breaking to maintain randomness within sparsity classes
    b = bytes(vector)
    h = zlib.crc32(b)
    
    # Scale zeros to dominate the priority
    return float(zeros) * 10000000.0 + float(h)
