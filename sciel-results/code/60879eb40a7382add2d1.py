def construct():
    n = 256
    S = set()
    
    # Check validity of adding x to S
    def can_add(x, S):
        # New triples involving x:
        # 1. a=x, b,c in S with x < b <= c
        # 2. a in S, b=x, c in S with a < x <= c (impossible if x is smallest new, but S has larger)
        #    Actually, if we add x to S, x could be a, b, or c.
        #    Since we iterate descending, S contains numbers > x.
        #    So x is the smallest element in S U {x}.
        #    Thus x can only be 'a'.
        #    So we only need to check: for all b, c in S with b <= c, x does not divide b+c.
        
        # Optimization: 
        # b+c ranges from 2*min(S) to 2*max(S).
        # We just iterate all pairs. Size of S is at most 256. Pairs ~ 32k. 
        # 256 * 32k = 8M ops. Might be slow in Python if done naively for each step.
        # But we only add if valid.
        
        for b in S:
            for c in S:
                if b > c:
                    continue
                # We need x < b. Since x < all in S, this is always true.
                if (b + c) % x == 0:
                    return False
        return True

    for x in range(n, 0, -1):
        if can_add(x, S):
            S.add(x)
            
    return sorted(S)
