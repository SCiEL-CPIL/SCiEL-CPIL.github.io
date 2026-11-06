def priority(vector, n):
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    
    # Prefer vectors with fewer 1s (1s are more likely to be AP midpoints)
    # Secondary: prefer vectors with more 0s (0s are "safer" endpoints)
    # Tertiary: deterministic hash for consistent tie-breaking
    
    h = 0
    for x in vector:
        h = (h * 31 + x + 1) & 0xFFFFFFFF
    
    # Primary: fewer 1s is better (higher priority)
    # Secondary: more 0s is better
    # Tertiary: hash for deterministic variation
    score = -c1 * 1000.0 + c0 * 10.0 + (h / 0xFFFFFFFF) * 1.0
    return score
