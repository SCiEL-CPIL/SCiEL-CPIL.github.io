def priority(vector, n):
    """
    Prioritizes vectors based on structural complexity (number of distinct symbols),
    preferring vectors with more distinct symbols (3) over simpler ones (2 or 1).
    Ties are broken using a high-precision pseudo-random hash to ensure a stable,
    diverse ordering within each complexity class.
    
    Hypothesis: Vectors with 3 distinct symbols are more "spread out" in the space
    and may form a better backbone for the cap set, allowing more total vectors
    to be included than a purely random or simple-first approach.
    """
    # 1. Determine structural complexity: number of distinct symbols
    # Since values are in {0, 1, 2}, we can use a bitmask or set.
    # For speed, manual check is often faster than set() for small n, 
    # but set() is fine and clean.
    # Optimization: Use a 3-bit mask.
    mask = 0
    for v in vector:
        mask |= (1 << v)
    
    # Count bits in mask
    # mask 1 (001) -> 1 distinct (0s)
    # mask 2 (010) -> 1 distinct (1s)
    # mask 4 (100) -> 1 distinct (2s)
    # mask 3 (011) -> 2 distinct (0,1)
    # mask 5 (101) -> 2 distinct (0,2)
    # mask 6 (110) -> 2 distinct (1,2)
    # mask 7 (111) -> 3 distinct (0,1,2)
    
    if mask == 1 or mask == 2 or mask == 4:
        distinct_count = 1
    elif mask == 3 or mask == 5 or mask == 6:
        distinct_count = 2
    else: # mask == 7
        distinct_count = 3
        
    # 2. Assign base score based on complexity
    # We prioritize Complex (3) > Medium (2) > Simple (1)
    # This is the structural change from the "Random" (current) and "Simple-First" (inspiration).
    base_score = float(distinct_count)
    
    # 3. Generate a pseudo-random tie-breaker
    # Encode vector as base-3 integer
    val = 0
    for v in vector:
        val = val * 3 + v
        
    # Apply SplitMix64 mixing
    val = (val + 0x9e3779b97f4a7c15) & 0xFFFFFFFFFFFFFFFF
    val = (val ^ (val >> 30)) * 0xbf58476d1ce4e5b9 & 0xFFFFFFFFFFFFFFFF
    val = (val ^ (val >> 27)) * 0x94d049bb133111eb & 0xFFFFFFFFFFFFFFFF
    val = val ^ (val >> 31)
    
    # Convert to a fraction in [0, 1) with 53-bit precision
    # This ensures that for n up to ~33, all vectors in the same class are uniquely ordered
    # without floating point precision loss affecting the class separation.
    # We take the top 53 bits of the 64-bit hash.
    random_frac = (val >> 11) / (1 << 53)
    
    # 4. Combine
    # The random_frac is strictly less than 1.0, so it won't cross into the next integer base_score.
    return base_score + random_frac
