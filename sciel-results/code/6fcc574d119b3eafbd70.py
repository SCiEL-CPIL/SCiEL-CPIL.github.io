def construct():
    # Try specific high-cardinality candidates
    candidates = [
        list(range(1, 3)) + list(range(66, 100)), # {1, 2, 66..99} size 36
        list(range(1, 4)) + list(range(66, 99)),  # {1, 2, 3, 66..98} size 36
    ]
    
    def has_subset_sum_200(nums):
        reachable = {0}
        for x in nums:
            new = {s + x for s in reachable if s + x <= 200}
            if 200 in new:
                return True
            reachable |= new
        return 200 in reachable
        
    for cand in candidates:
        if not has_subset_sum_200(cand):
            return cand
            
    # Existing logic...
