def construct():
    S = []
    for i in range(256, 0, -1):
        can_add = True
        # Check 1: i is 'a'
        # Need i not divide b+c for all b,c in S with b > i
        # Since S only has elements > i (iterating down), all b,c in S are > i.
        for j in range(len(S)):
            for k in range(j, len(S)):
                b = S[j]
                c = S[k]
                if (b + c) % i == 0:
                    can_add = False
                    break
            if not can_add:
                break
        
        if can_add:
            # Check 2: i is 'b'
            # Need a not divide i+c for all a in S with a < i, c in S with c >= i
            # a < i is always true for a in S (since S has > i? No, S has elements we added previously, which are > i).
            # Wait, S contains elements GREATER than i.
            # So a > i. But condition is a < b=i.
            # So there are NO a in S such that a < i.
            # So Check 2 is vacuously true?
            # Yes, because we are adding i to a set of numbers all larger than i.
            # So i is the smallest element in S_new.
            # So i can only be 'a'.
            
            # Check 3: i is 'c'
            # Need a not divide b+i for all a < b <= i in S_new.
            # a, b must be in S (since i is new).
            # But S has elements > i. So a > i, b > i.
            # Condition a < b <= i.
            # Since a > i, a < i is false.
            # So Check 3 is vacuously true?
            
            # So if we iterate downwards, i is always the minimum of S_new.
            # So we only need to check if i divides any sum of two existing elements.
            
            if can_add:
                S.append(i)
    
    return S
