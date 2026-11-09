def priority(vector, n):
    # Count 1s
    count_1 = 0
    for v in vector:
        if v == 1:
            count_1 += 1
    
    # Deterministic hash to break ties
    # Simple hash: sum of (value * prime^index)
    h = 0
    base = 31
    for i, v in enumerate(vector):
        h = (h * base + v) % 1000000007
    
    # Normalize hash to [0, 1)
    hash_val = h / 1000000007.0
    
    # Priority: lower count_1 is better (higher score)
    # We subtract count_1, so vectors with 0 ones get ~1.0, 1 one gets ~0.0, etc.
    # Wait, if count_1 is 0, score is ~1.0. If count_1 is 1, score is ~0.0.
    # This creates a large gap.
    # Better: score = -count_1 + hash_val
    return -count_1 + hash_val
