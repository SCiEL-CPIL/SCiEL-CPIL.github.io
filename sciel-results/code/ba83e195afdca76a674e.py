def construct():
    S = set()
    # We iterate in descending order to prioritize large numbers
    # which have fewer valid pair sums within the bound [1, 128]
    for x in range(128, 0, -1):
        # Find all y in S such that x + y is also in S
        # Note: x + y must be <= 128 to be in S
        # Since y >= 1, x + y > x.
        # We only care about y such that x + y <= 128
        candidates = []
        for y in S:
            if x + y <= 128 and (x + y) in S:
                candidates.append(y)
        
        # Check if there exist distinct a, b in candidates
        # such that a + b is in S OR a + b == x
        valid = True
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                a = candidates[i]
                b = candidates[j]
                if a == b:
                    continue
                sum_ab = a + b
                if sum_ab in S or sum_ab == x:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            S.add(x)
            
    return [int(x) for x in S]
