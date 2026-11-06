def priority(vector, n):
    """
    Prioritizes vectors using a deterministic pseudo-random permutation of the hypercube.
    This approach avoids local clustering biases inherent in structural heuristics (like
    zero-count or distinct-symbol counts), allowing the greedy search to explore the
    space more uniformly. The hypothesis is that a uniform ordering prevents the
    premature selection of vectors that block large portions of the remaining space,
    potentially leading to a larger final cap set.
    """
    # Encode vector as base-3 integer
    val = 0
    for v in vector:
        val = val * 3 + v
        
    # Apply SplitMix64 mixing for good distribution
    # This creates a high-quality hash that acts as a uniform random priority
    val = (val + 0x9e3779b97f4a7c15) & 0xFFFFFFFFFFFFFFFF
    val = (val ^ (val >> 30)) * 0xbf58476d1ce4e5b9 & 0xFFFFFFFFFFFFFFFF
    val = (val ^ (val >> 27)) * 0x94d049bb133111eb & 0xFFFFFFFFFFFFFFFF
    val = val ^ (val >> 31)
    
    # Convert to fraction [0, 1)
    # Using higher bits for precision
    random_frac = (val >> 11) / (1 << 53)
    
    return random_frac
