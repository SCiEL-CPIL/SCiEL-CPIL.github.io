def priority(vector, n):
    """Return a priority float for a vector in {0,1,2}^n.
    Vectors with fewer 1s are prioritized first, because the set {0,2}^n
    (vectors with no 1s) is itself a cap set of size 2^n. By ordering these
    first, the evaluator's greedy cap-set build captures a large base cap set,
    then fills in remaining vectors that do not create 3-term arithmetic
    progressions. A hash-based tie-breaker adds diversity within each
    ones-count bucket.
    """
    ones = 0
    for x in vector:
        if x == 1:
            ones += 1
    # Deterministic hash for tie-breaking
    idx = 0
    for x in vector:
        idx = idx * 3 + x
    h = idx
    h ^= (h >> 16)
    h = (h * 0x85ebca6b) & 0xFFFFFFFF
    h ^= (h >> 13)
    h = (h * 0xc2b2ae35) & 0xFFFFFFFF
    h ^= (h >> 16)
    hash_frac = (h & 0x7FFFFFFF) / 0x7FFFFFFF
    # Normalize ones count to [0,1]
    ones_frac = ones / n if n > 0 else 0.0
    # Strongly favor fewer 1s; hash breaks ties within a bucket.
    return (1.0 - ones_frac) * 10.0 + hash_frac
