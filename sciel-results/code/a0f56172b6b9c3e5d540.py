def construct():
    n = 256
    S = set()
    
    # To maximize cardinality, we should probably try to include as many numbers as possible.
    # Greedy from n down to 1 is a standard heuristic for such independence-like problems.
    for x in range(n, 0, -1):
        can_add = True
        # Check if adding x creates a violation
        # Violation: exists a, b, c in S U {x} such that a < b <= c and a | (b+c)
        
        # We only need to check triples involving x.
        # Case 1: x is the smallest (a). x < b <= c.
        # b, c must be in S and > x.
        # Check if x divides b+c for any b,c in S with x < b <= c.
        larger_in_S = [y for y in S if y > x]
        for i in range(len(larger_in_S)):
            for j in range(i, len(larger_in_S)):
                b = larger_in_S[i]
                c = larger_in_S[j]
                # Since S is built from large to small, all y in S are > x? 
                # No, we are iterating down. So S contains numbers > x.
                # So b, c are in S and > x.
                if (b + c) % x == 0:
                    can_add = False
                    break
            if not can_add:
                break
        
        if not can_add:
            continue
            
        # Case 2: x is the middle (b). a < x <= c.
        # a, c in S. a < x <= c.
        # Since S contains only numbers > x (from previous iterations), 
        # wait. We iterate from n down to 1.
        # When we consider x, S contains {y | y > x and y was accepted}.
        # So all elements in S are strictly greater than x.
        # Therefore, x cannot be 'b' if 'a' must be in S and a < x. 
        # Because S has no elements < x.
        # So Case 2 (x=b) and Case 3 (x=c) where a,b in S and a < b <= x 
        # implies a,b < x? No, b=x. a < x. But S has no elements < x.
        # So a cannot be in S.
        # Thus, the ONLY potential violations when adding x (where x is the new smallest element) 
        # are those where x is the divisor 'a'.
        
        # Let's re-verify.
        # S_current = {y | y > x, accepted}.
        # We add x. New set S_new = S_current U {x}.
        # Any triple (a,b,c) in S_new with a<b<=c and a | b+c.
        # If a, b, c are all in S_current, they were already checked.
        # If one of them is x:
        #   If x=c: a < b <= x. Since a,b in S_current, a,b > x. Contradiction (a > x and a < x).
        #   If x=b: a < x <= c. a in S_current => a > x. Contradiction (a < x and a > x).
        #   If x=a: x < b <= c. b,c in S_current => b,c > x. This is possible.
        
        # So we ONLY need to check if x divides b+c for any b,c in S_current with x < b <= c.
        
        # My loop above does exactly this.
        
        if can_add:
            S.add(x)
            
    return list(S)
