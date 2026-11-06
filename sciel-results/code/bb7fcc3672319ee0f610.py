def _solve_for_M_bitmask(M, deadline):
    # A is bitmask
    # used_diffs is bitmask
    # missing is bitmask (bits 1..30)
    
    initial_A = (1 << 0) | (1 << M)
    initial_used = (1 << M)
    initial_missing = 0
    for d in range(1, 31):
        if d != M:
            initial_missing |= (1 << d)
            
    # Precompute diffs for all pairs?
    # No.
    
    def search(A, used, missing, depth=0):
        if time.time() > deadline:
            return None
        
        if missing == 0:
            # Convert bitmask to list
            res = []
            for i in range(M+1):
                if A & (1 << i):
                    res.append(i)
            return res
        
        if bin(A).count("1") > 32:
            return None
            
        candidates = []
        
        for x in range(1, M):
            if A & (1 << x):
                continue
            
            new_diffs = []
            valid = True
            # Iterate over set bits in A
            temp_A = A
            while temp_A:
                # Get lowest set bit
                lsb = temp_A & -temp_A
                a = lsb.bit_length() - 1
                d = abs(x - a)
                if used & (1 << d):
                    valid = False
                    break
                if d in new_diffs:
                    valid = False
                    break
                new_diffs.append(d)
                temp_A ^= lsb
            
            if not valid:
                continue
            
            coverage = 0
            for d in new_diffs:
                if missing & (1 << d):
                    coverage += 1
            
            candidates.append((coverage, x, new_diffs))
        
        if not candidates:
            return None
        
        candidates.sort(key=lambda item: (-item[0], item[1]))
        
        for coverage, x, new_diffs in candidates[:10]:
            new_A = A | (1 << x)
            
            new_used = used
            for d in new_diffs:
                new_used |= (1 << d)
            
            new_missing = missing
            for d in new_diffs:
                new_missing &= ~(1 << d)
            
            res = search(new_A, new_used, new_missing, depth + 1)
            if res is not None:
                return res
        
        return None
    
    return search(initial_A, initial_used, initial_missing)
