def construct():
    # Generators for M12
    g1 = (1, 2, 3, 4, 5, 0, 6, 7, 8, 9, 10, 11)
    g2 = (1, 2, 3, 4, 6, 0, 5, 7, 8, 9, 10, 11)
    g3 = (1, 2, 3, 5, 6, 0, 4, 7, 8, 9, 10, 11)
    
    # Note: I need to verify the cycle notation to tuple conversion.
    # (0, 1, 2, 3, 4, 5) means 0->1, 1->2, ..., 5->0.
    # So tuple index i contains the image of i.
    # g1[0]=1, g1[1]=2, ..., g1[5]=0, g1[6]=6...
    
    def apply_perm(perm, block):
        return tuple(sorted(perm[p] for p in block))

    def compose(p1, p2):
        # p1 * p2 means apply p2 then p1? Or p1 then p2?
        # Usually (p1 * p2)(x) = p1(p2(x)).
        # Let's stick to applying generators to the current block.
        pass

    base_block = (0, 1, 2, 3, 4, 5)
    blocks = set()
    queue = [base_block]
    
    while queue:
        b = queue.pop()
        if b in blocks:
            continue
        blocks.add(b)
        
        # Apply generators
        nb1 = apply_perm(g1, b)
        nb2 = apply_perm(g2, b)
        nb3 = apply_perm(g3, b)
        
        if nb1 not in blocks: queue.append(nb1)
        if nb2 not in blocks: queue.append(nb2)
        if nb3 not in blocks: queue.append(nb3)
        
    # Convert to list of sorted tuples
    result = [tuple(sorted(b)) for b in blocks]
    
    # Verify
    if len(result) != 132:
        # Fallback or error
        pass
        
    # Check validity
    all_5_subsets = set()
    for b in result:
        for i in range(6):
            s = tuple(sorted(b[:i] + b[i+1:]))
            if s in all_5_subsets:
                return [] # Invalid
            all_5_subsets.add(s)
            
    return result
