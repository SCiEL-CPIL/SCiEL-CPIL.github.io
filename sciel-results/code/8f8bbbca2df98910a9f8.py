import time

def construct():
    start_time = time.time()
    deadline = 110  # Safety margin
    
    from itertools import combinations
    
    # Precompute bitmasks for all 5-subsets? 
    # No, we generate them on the fly or precompute mapping from 5-subset tuple to bitmask.
    # Actually, we just need to check if a 5-subset is covered.
    # We can store covered 5-subsets as a set of integers (bitmasks).
    
    covered = set()
    blocks = []
    
    # Iterate through all 6-subsets in lexicographic order
    for b in combinations(range(12), 6):
        if time.time() > deadline:
            break
            
        # Generate the 6 bitmasks for the 5-subsets of b
        # b is a tuple of 6 ints
        # 5-subsets are obtained by removing one element
        # We need to check if ANY of these 5-subsets is already in 'covered'
        
        # Optimization: 
        # If we represent a 6-subset as a 12-bit int, we can derive the 5-subsets.
        # But iterating 6 subsets is cheap.
        
        conflict = False
        sub_bitmasks = []
        
        # b has 6 elements. We remove each one to get a 5-subset.
        # Example: b = (0,1,2,3,4,5)
        # Remove 0 -> (1,2,3,4,5) -> mask
        # Remove 1 -> (0,2,3,4,5) -> mask
        # ...
        
        # To avoid recalculating the mask from scratch for each, 
        # we can compute the full mask of b, then XOR out the bit for the removed element.
        full_mask = 0
        for x in b:
            full_mask |= (1 << x)
            
        for x in b:
            sub_mask = full_mask ^ (1 << x)
            if sub_mask in covered:
                conflict = True
                break
            sub_bitmasks.append(sub_mask)
            
        if not conflict:
            blocks.append(list(b))
            covered.update(sub_bitmasks)
            
    return blocks
