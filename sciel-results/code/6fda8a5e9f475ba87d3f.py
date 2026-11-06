def construct():
    """
    Construct a subset of [1, 100] with no subset summing to 200.
    Uses a greedy algorithm that iterates from 100 down to 1.
    Adds an element x if adding it does not create a subset sum of 200.
    This is valid because:
    If S is a set with no subset summing to 200, and 200-x is NOT in the subset sums of S,
    then S U {x} also has no subset summing to 200.
    Any new subset sum involving x is of the form s + x where s is a subset sum of S.
    If s + x = 200, then s = 200 - x, which we checked is not in S's sums.
    """
    # Bitset of reachable sums.
    # Index i is 1 if sum i is reachable.
    # Max sum is 100*101/2 = 5050, but we only care up to 200.
    # However, sums can exceed 200, but if a sum exceeds 200, it can't contribute to 200.
    # We only need to track sums <= 200.
    # Let's use a set or a boolean array of size 201.
    
    reachable = [False] * 201
    reachable[0] = True
    
    selected = []
    
    for x in range(100, 0, -1):
        # Check if adding x creates a sum of 200.
        # This happens if there exists a sum s in reachable such that s + x == 200.
        # i.e., s == 200 - x.
        
        if 200 - x < 0:
            # x > 200, always safe, but x <= 100 here.
            pass
        
        if 200 - x <= 200:
            if reachable[200 - x]:
                # Adding x would create a sum of 200. Skip x.
                continue
        
        # Safe to add x.
        selected.append(x)
        
        # Update reachable sums.
        # New sums are old sums and old sums + x.
        # We must update from high to low to avoid using x multiple times (0/1 knapsack style).
        # But wait, we are just checking if 200 becomes reachable.
        # We need to update the `reachable` array for future steps.
        
        # Iterate s from 200 down to 0
        # If reachable[s] is True, then reachable[s+x] becomes True (if s+x <= 200)
        for s in range(200, -1, -1):
            if reachable[s]:
                new_sum = s + x
                if new_sum <= 200:
                    reachable[new_sum] = True
                else:
                    # Sum exceeds 200, we don't care about it for the "200" check,
                    # but do we care for future additions?
                    # If a sum exceeds 200, it can never be part of a subset summing to exactly 200
                    # because all elements are positive.
                    # So we can ignore sums > 200.
                    pass
    
    # The selected list is the result.
    # The order doesn't matter for the return value, just the set.
    return selected
