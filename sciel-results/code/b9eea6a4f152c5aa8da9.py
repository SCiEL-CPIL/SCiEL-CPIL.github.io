def construct():
    n = 128
    L = set(range(65, n + 1))
    T = set()
    
    def is_valid_add(x, T):
        S = T | {x}
        # Check triples involving x as a member
        # (x, u, v) with u, v in T
        smalls = list(T)
        for i in range(len(smalls)):
            for j in range(i+1, len(smalls)):
                u = smalls[i]
                v = smalls[j]
                # Check if (x, u, v) is a forbidden triple
                if (x + u in S) and (x + v in S) and (u + v in S):
                    return False
        
        # Check triples where x is a sum
        # (u, v, w) in T such that one sum is x
        # WLOG u + v = x
        # Then we need u + w in S and v + w in S
        # u + w in S implies u + w in T or u + w == x
        # If u + w == x, then w = x - u = v, but w != v. So u + w must be in T.
        # Similarly v + w must be in T.
        
        # Find pairs in T summing to x
        # Since T is small, just iterate
        for i in range(len(smalls)):
            for j in range(i+1, len(smalls)):
                u = smalls[i]
                v = smalls[j]
                if u + v == x:
                    # Check if there exists w in T \ {u, v} such that u+w in T and v+w in T
                    for k in range(len(smalls)):
                        if k == i or k == j:
                            continue
                        w = smalls[k]
                        if (u + w in T) and (v + w in T):
                            return False
        return True

    candidates = list(range(1, 65))
    # Try descending order
    for x in sorted(candidates, reverse=True):
        if is_valid_add(x, T):
            T.add(x)
            
    return sorted(T | L)
