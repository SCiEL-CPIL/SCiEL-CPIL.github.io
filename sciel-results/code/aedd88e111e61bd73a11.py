def construct():
    # Target sum
    target = 200
    # Elements
    elements = list(range(1, 101))
    elements.sort(reverse=True)
    
    result = []
    # Bitset of achievable sums
    # Using a set or a boolean array of size target+1
    achievable = [False] * (target + 1)
    achievable[0] = True
    
    for x in elements:
        # Check if adding x creates a sum of target
        # We need to know if there is a subset of current result summing to target - x
        # If target - x < 0, skip check (x > target, not possible here)
        if x <= target:
            need = target - x
            if need >= 0 and achievable[need]:
                # Adding x would complete a sum of target
                # So we skip x
                continue
        
        # Add x to result
        result.append(x)
        
        # Update achievable sums
        # We iterate backwards to avoid using x multiple times in the same step?
        # No, we are adding x to the set of available numbers.
        # New sums = Old sums + x
        # Since we process x once, we update the bitset.
        # We must be careful: if we update in place, we might use x twice?
        # No, we are just marking sums achievable by the set {result}.
        # The standard knapsack update is:
        # for s in range(target, x-1, -1):
        #     if achievable[s-x]:
        #         achievable[s] = True
        
        for s in range(target, x - 1, -1):
            if achievable[s - x]:
                achievable[s] = True

    if not result:
        return [1]
        
    return result
