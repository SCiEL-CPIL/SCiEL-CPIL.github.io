def _greedy_build(order):
    selected = []
    sums = set()
    
    for x in order:
        # Check if x can be appended
        # New sums: x, c[-1]+x, c[-2]+c[-1]+x, ...
        
        # 1. Check x
        if x in sums:
            continue
            
        # 2. Check suffixes
        valid = True
        suffix_sum = 0
        # We need to store the suffix sums if valid to add them later?
        # Or just recompute? Recomputing is O(n). Storing is O(n) memory.
        # Let's recompute to keep memory low, as n is small (< 20).
        
        for i in range(len(selected) - 1, -1, -1):
            suffix_sum += selected[i]
            total = suffix_sum + x
            if total in sums:
                valid = False
                break
        
        if valid:
            selected.append(x)
            sums.add(x)
            # Add the suffix sums
            suffix_sum = 0
            for i in range(len(selected) - 2, -1, -1):
                suffix_sum += selected[i]
                sums.add(suffix_sum + x)
                
    return selected
