def construct():
    n = 256
    S = set()
    
    # Greedy from largest to smallest
    for x in range(n, 0, -1):
        can_add = True
        # Check constraints with existing elements in S
        # Since we add in descending order, all y in S are > x
        for y in S:
            # Constraint 1: x does not divide 2y
            if (2 * y) % x == 0:
                can_add = False
                break
            
            # Constraint 2: x does not divide y + z for any z in S, z >= y
            # We need to check all pairs (y, z) in S
            # To avoid O(N^3), we can precompute or just iterate.
            # N is 256, S size is ~100. 100*100 = 10k checks per x. 256 * 10k = 2.5M ops. Fast.
            for z in S:
                if z < y:
                    continue
                if (y + z) % x == 0:
                    can_add = False
                    break
            if not can_add:
                break
        
        if can_add:
            S.add(x)
            
    return list(S)
