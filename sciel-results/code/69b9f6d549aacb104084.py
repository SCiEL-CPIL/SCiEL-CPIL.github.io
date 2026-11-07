<<<<<<< SEARCH
def _try_add_bitmask(reach, x):
    """
    Try to add x to the set represented by reach (list of bitmasks).
    reach[k] is bitmask of sums with exactly k elements.
    Returns new_reach (list of bitmasks) if valid, else None.
    """
    n = len(reach) - 1
    new_reach = [0] * (n + 2)
    
    # Copy old reach
    for k in range(n + 1):
        new_reach[k] = reach[k]
        
    # Add new sums
    for k in range(1, n + 2):
        new_reach[k] |= (reach[k-1] << x)
        
    # Check pairwise disjointness for new_reach
    # Optimization: Only check pairs involving the "new" parts?
    # No, just check all pairs. N is small.
    for i in range(n + 2):
        for j in range(i + 1, n + 2):
            if new_reach[i] & new_reach[j]:
                return None
                
    return new_reach
=======
def _try_add_bitmask(reach, x):
    """
    Try to add x to the set represented by reach (list of bitmasks).
    reach[k] is bitmask of sums with exactly k elements.
    Returns new_reach (list of bitmasks) if valid, else None.
    """
    n = len(reach) - 1
    new_reach = [0] * (n + 2)
    
    # Copy old reach
    for k in range(n + 1):
        new_reach[k] = reach[k]
        
    # Compute new sums (the bits added to each cardinality level)
    new_sums = [0] * (n + 2)
    for k in range(1, n + 2):
        new_sums[k] = reach[k-1] << x
        new_reach[k] |= new_sums[k]
        
    # Check disjointness: for each pair (i,j), check if any new bit in new_sums[i]
    # conflicts with old bits in reach[j] or new bits in new_sums[j]
    # Optimization: check new_sums[i] against (reach[j] | new_sums[j]) for j != i
    # and new_sums[i] & new_sums[j] for i < j
    for i in range(n + 2):
        if new_sums[i] == 0:
            continue
        # Check against all other levels
        for j in range(n + 2):
            if i == j:
                continue
            # new bits at level i must not conflict with old bits at level j
            if new_sums[i] & reach[j]:
                return None
            # new bits at level i must not conflict with new bits at level j (only check i<j to avoid double)
            if i < j and new_sums[i] & new_sums[j]:
                return None
                
    return new_reach
>>>>>>> REPLACE

