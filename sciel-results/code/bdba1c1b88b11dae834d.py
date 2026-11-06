def priority(vector, n):
    """
    Prioritizes vectors based on the number of zero entries (Hamming weight complement),
    with secondary structural features capturing variance and balance of non-zero entries.
    Hypothesis: Vectors with more zeros are 'sparser' and less likely to form
    3-term arithmetic progressions. Among vectors with equal zero counts, those
    with lower variance in non-zero entries may have more constrained AP completions.
    """
    import time
    
    # 1. Count zeros, ones, and twos; compute sum of squares
    zero_count = 0
    one_count = 0
    two_count = 0
    sum_sq = 0
    for v in vector:
        if v == 0:
            zero_count += 1
        elif v == 1:
            one_count += 1
            sum_sq += 1
        else:  # v == 2
            two_count += 1
            sum_sq += 4
    
    # 2. Base score: Higher zero count is primary.
    base_score = float(zero_count)
    
    # 3. Secondary: variance of non-zero entries.
    # In F_3, a 3-AP condition involves x+z=2y mod 3.
    # For fixed zero count, vectors with non-zero entries clustered around
    # a single value have more uniform constraints per coordinate,
    # potentially limiting AP completions. We reward lower variance.
    non_zero = one_count + two_count
    if non_zero > 0:
        mean_val = sum_sq / non_zero  # This is E[v^2], not E[v], but monotonic
        # Variance proxy: E[v^2] - (E[v])^2, but we only have E[v^2] easily
        # Use sum_sq / non_zero as a proxy for "spread" - lower is more uniform
        # Normalize to [0, 1]: min sum_sq/non_zero = 1 (all ones), max = 4 (all twos)
        spread = (sum_sq / non_zero - 1.0) / 3.0  # 0 for all-ones, 1 for all-twos
        # Reward lower spread (more uniform non-zero entries)
        variance_bonus = 1.0 - spread
    else:
        variance_bonus = 1.0  # All zeros
    
    # 4. Tertiary: balance between 1s and 2s (weaker)
    if non_zero > 0:
        imbalance = abs(one_count - two_count) / non_zero
        balance_bonus = 1.0 - imbalance
    else:
        balance_bonus = 1.0
    
    # Add secondary terms with small weights to not override zero count
    base_score += 0.08 * variance_bonus + 0.04 * balance_bonus
    
    # 5. Generate a pseudo-random tie-breaker
    # Encode vector as base-3 integer
    val = 0
    for v in vector:
        val = val * 3 + v
        
    # Apply SplitMix64 mixing for good distribution
    val = (val + 0x9e3779b97f4a7c15) & 0xFFFFFFFFFFFFFFFF
    val = (val ^ (val >> 30)) * 0xbf58476d1ce4e5b9 & 0xFFFFFFFFFFFFFFFF
    val = (val ^ (val >> 27)) * 0x94d049bb133111eb & 0xFFFFFFFFFFFFFFFF
    val = val ^ (val >> 31)
    
    # Convert to fraction [0, 1)
    random_frac = (val >> 11) / (1 << 53)
    
    # 6. Combine
    return base_score + random_frac
