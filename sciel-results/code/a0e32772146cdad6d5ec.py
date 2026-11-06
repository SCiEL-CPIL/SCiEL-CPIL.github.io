def construct():
    n = 256
    S = []
    # Try adding numbers from n down to 1
    for x in range(n, 0, -1):
        valid = True
        
        # Case 1: x is the smallest element (a)
        # Check all pairs (b, c) in S such that x < b <= c
        # Since S is built from larger numbers, all elements in S are > x?
        # Not necessarily, if we skipped some. But we iterate n down to 1.
        # So all elements in S are > x.
        # Thus x < b is always true for b in S.
        # We just need b <= c.
        for i in range(len(S)):
            for j in range(i, len(S)):
                b = S[i]
                c = S[j]
                # b and c are from S. Since we iterate descending, S is not sorted.
                # Let's sort S or just check all pairs.
                # Actually, if I just check all pairs (b, c) in S with b <= c,
                # and since all elements in S are > x, the condition x < b is satisfied.
                if (b + c) % x == 0:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            # Case 2: x is the middle element (b)
            # Check all pairs (a, c) in S such that a < x <= c
            for a in S:
                if a < x:
                    for c in S:
                        if x <= c:
                            if (x + c) % a == 0:
                                valid = False
                                break
                if not valid:
                    break
            
            # Case 3: x is the largest element (c)
            # Check all pairs (a, b) in S such that a < b <= x
            if valid:
                for a in S:
                    if not valid:
                        break
                    for b in S:
                        if a < b and b <= x:
                            if (b + x) % a == 0:
                                valid = False
                                break
        
        if valid:
            S.append(x)
            
    return S
