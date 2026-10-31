# EVOLVE-BLOCK-START
import zlib

def priority(vector, n):
    # Prioritize vectors with a balanced distribution of 0s, 1s, and 2s,
    # with a slight penalty for 1s (midpoint symbol in affine lines).
    # Add hash-based tie-breaking to break symmetries.
    n1 = 0
    n2 = 0
    for x in vector:
        if x == 1:
            n1 += 1
        elif x == 2:
            n2 += 1
    n0 = n - n1 - n2
    
    target = n / 3.0
    # Weighted deviation: penalize 1s slightly more since they are midpoints
    deviation = (n0 - target)**2 + 1.1 * (n1 - target)**2 + (n2 - target)**2
    
    # Hash-based tie-breaking for deterministic but varied ordering
    # Convert tuple to bytes for crc32
    h = zlib.crc32(bytes(vector))
    # Scale hash to a small perturbation that doesn't override the main heuristic
    hash_component = (h % 10000) / 10000.0 * 0.01
    
    return -deviation + hash_component
# EVOLVE-BLOCK-END
