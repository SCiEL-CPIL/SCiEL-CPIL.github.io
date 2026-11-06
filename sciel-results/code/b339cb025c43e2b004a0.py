def construct():
    from math import lcm
    import random
    
    # Precompute conflicts
    # conflicts[x] is a set of y such that lcm(x, y) <= 256
    max_val = 256
    candidates = list(range(2, max_val + 1))
    conflicts = {x: set() for x in candidates}
    
    for i, x in enumerate(candidates):
        for y in candidates[i+1:]:
            if lcm(x, y) <= max_val:
                conflicts[x].add(y)
                conflicts[y].add(x)
                
    # Greedy initialization
    # Sort by 1/x descending (i.e., x ascending)
    # Actually, larger 1/x is better. Smaller x has larger 1/x.
    # So we prefer small x.
    # But small x conflict with many things.
    # Let's try greedy: sort by 1/x descending.
    selected = []
    for x in sorted(candidates, key=lambda v: 1.0/v, reverse=True):
        # Check if x is compatible with all in selected
        # i.e., x is not in conflicts[y] for any y in selected
        # Or simply: no y in selected such that x in conflicts[y]
        # Since conflicts is symmetric, x in conflicts[y] == y in conflicts[x]
        compatible = True
        for y in selected:
            if x in conflicts[y]:
                compatible = False
                break
        if compatible:
            selected.append(x)
            
    # Local Search
    # Try to swap one element out and one in, or just add one in if possible?
    # Since we maximize sum, we might want to replace a large x (small weight) with a smaller x (large weight) if it fits.
    # Or replace a set of elements?
    
    # Let's do a simple 1-swap local search first, but optimized.
    # State: set of selected elements.
    # Weight: sum(1/x)
    
    current_set = set(selected)
    current_sum = sum(1.0/x for x in current_set)
    
    improved = True
    while improved:
        improved = False
        # Try to add any element not in set
        for x in candidates:
            if x in current_set:
                continue
            # Check if x is compatible with all in current_set
            # i.e., current_set is disjoint from conflicts[x]
            if not (current_set & conflicts[x]):
                current_set.add(x)
                current_sum += 1.0/x
                improved = True
                break # Restart loop to check more additions
                
        if improved:
            continue
            
        # Try to swap: remove y, add x
        # To improve sum, we need 1/x > 1/y, so x < y.
        # We iterate x (new) in increasing order (high weight).
        # We iterate y (old) in increasing order (low weight is good to remove? No, we remove to free up space).
        # Actually, we want to replace a "bad" element (low weight, high conflict) with a "good" one.
        
        for x in sorted(candidates, key=lambda v: 1.0/v, reverse=True):
            if x in current_set:
                continue
            # If we add x, we must remove all y in current_set that conflict with x.
            # Let R = {y in current_set | y in conflicts[x]}
            # We can only add x if we are willing to remove R.
            # The net change in sum is 1/x - sum(1/y for y in R).
            # We want this to be > 0.
            # Also, after removing R, is x compatible with remaining? Yes, by definition of R.
            # Are remaining elements compatible with each other? Yes, they were before.
            
            # However, simply removing R might be bad if R has high total weight.
            # We can try to replace a subset of R? No, if x conflicts with y, y MUST be removed.
            # So the only option is to remove ALL conflicts of x in the current set.
            
            R = current_set & conflicts[x]
            if not R:
                # This case is handled by the "add" loop above, but if we didn't break, maybe we missed it?
                # The "add" loop breaks on first success. If no add was possible, then for all x not in set, R is not empty.
                pass
            
            # Calculate gain
            weight_x = 1.0 / x
            weight_R = sum(1.0 / y for y in R)
            
            if weight_x > weight_R + 1e-12:
                # Perform swap
                current_set.difference_update(R)
                current_set.add(x)
                current_sum += weight_x - weight_R
                improved = True
                break # Restart to check for more adds or swaps
                
    return list(current_set)
