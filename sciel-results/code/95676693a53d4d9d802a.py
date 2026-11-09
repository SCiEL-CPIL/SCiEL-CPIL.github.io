import zlib
import time

# EVOLVE-BLOCK-START
_START_TIME = time.monotonic()
_DEADLINE = _START_TIME + 320.0

def priority(vector, n):
    if time.monotonic() > _DEADLINE:
        return 0.0

    # Structural change: Prioritize vectors with high Hamming weight (many non-zeros).
    # Hypothesis: Low-weight vectors are more likely to be part of short APs
    # because their sums are also low-weight, creating dense clusters of APs.
    # High-weight vectors are "spread out" and less likely to collide.
    
    weight = 0
    for v in vector:
        if v != 0:
            weight += 1
            
    # Use a high-quality hash for tie-breaking among vectors with the same weight.
    # This ensures a deterministic but "random" ordering within weight classes,
    # which helps avoid systematic biases in the greedy selection.
    h = zlib.crc32(bytes(vector))
    
    # Scale weight to dominate the hash tie-breaker.
    # Max weight is n. Max hash is 2^32.
    # We want weight to be the primary key.
    score = weight * (2**32) + h
    
    # Return as float for the interface
    return float(score)
# EVOLVE-BLOCK-END
