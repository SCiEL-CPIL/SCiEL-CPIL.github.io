def construct():
    candidates = list(range(256, 1, -1))
    S = []
    for x in candidates:
        can_add = True
        # Check 1: x does not divide b*c for any distinct b,c in S
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                b = S[i]
                c = S[j]
                if (b * c) % x == 0:
                    can_add = False
                    break
            if not can_add:
                break
        
        if can_add:
            # Check 2: No existing b divides x*c for any c in S \ {b}
            for b in S:
                for c in S:
                    if b == c:
                        continue
                    if b == x:
                        continue
                    # We are adding x, so the triple is (b, x, c) or (b, c, x) etc.
                    # Condition: a | b*c.
                    # If a=b, we check b | x*c.
                    if (x * c) % b == 0:
                        can_add = False
                        break
                if not can_add:
                    break
        
        if can_add:
            S.append(x)
            
    return S
