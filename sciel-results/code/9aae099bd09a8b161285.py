def construct():
    """
    Finds the longest contiguous interval [start, start + length - 1] within [1, 1008]
    such that for all x, y in the interval, x * y % 1009 != 1.
    """
    p = 1009
    n = 1008
    
    # Precompute modular inverses
    # inv[x] is the unique y in [1, p-1] such that x*y % p == 1
    inv = [0] * (n + 1)
    for x in range(1, n + 1):
        inv[x] = pow(x, p - 2, p)
        
    best_start = 1
    best_length = 0
    
    # Iterate over all possible start positions
    for s in range(1, n + 1):
        # For a fixed start s, we want to find the maximum e such that
        # for all x in [s, e], inv[x] is not in [s, e].
        # This is equivalent to e < min{ max(x, inv[x]) : x >= s, inv[x] >= s }
        
        min_threshold = n + 1  # Initialize to a value larger than any possible threshold
        
        # Check all x >= s
        for x in range(s, n + 1):
            ix = inv[x]
            if ix >= s:
                # Both x and ix are >= s.
                # If both are in [s, e], they form a conflict.
                # So e must be < max(x, ix).
                threshold = max(x, ix)
                if threshold < min_threshold:
                    min_threshold = threshold
                    # If the threshold is <= s, then e < s, so no valid interval
                    # of length >= 1 exists for this start s.
                    if min_threshold <= s:
                        break
        
        if min_threshold > s:
            # The maximum valid end is min_threshold - 1
            max_valid_e = min_threshold - 1
            length = max_valid_e - s + 1
            if length > best_length:
                best_length = length
                best_start = s
        else:
            # No valid interval starting at s (length 0)
            pass

    # Fallback if no valid interval found (should not happen as single elements like 2 are valid)
    if best_length == 0:
        best_start = 2
        best_length = 1

    return [best_start, best_length]
