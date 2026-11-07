import time

P = { ... } # From prompt

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency masks
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    best_distinct = 0
    best_mask = 0
    
    # Iterate all subsets
    # We can optimize by only considering masks with popcount >= 8
    # But iterating all 2^17 is fine.
    for mask in range(1, 1 << n):
        k = mask.bit_count()
        if k < 8:
            continue
            
        # Calculate degrees
        deg_set = set()
        # Iterate through vertices
        # Efficient way: while m:
        #   lsb = m & -m
        #   v = lsb.bit_length() - 1
        #   deg = (adj_mask[v] & mask).bit_count()
        #   deg_set.add(deg)
        #   m ^= lsb
        
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            deg = (adj_mask[v] & mask).bit_count()
            deg_set.add(deg)
            m ^= lsb
            
        distinct = len(deg_set)
        if distinct > best_distinct:
            best_distinct = distinct
            best_mask = mask
            # Theoretical max for this k is k. 
            # If distinct == k, it's max for this size.
            # But we want global max. 
            # Max possible distinct degrees in any graph of size 17 is 17.
            # If best_distinct == 17, we can break.
            if best_distinct == n:
                break
                
    # Convert best_mask to list of vertices
    result = []
    for v in range(n):
        if best_mask & (1 << v):
            result.append(v)
            
    return result
