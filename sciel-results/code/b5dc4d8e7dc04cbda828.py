def construct():
    # 1. Greedy Construction: Numbers with no digit 10 in base 11
    S = set()
    for x in range(1, 257):
        # Check if x has no digit 10 in base 11
        temp = x
        valid = True
        while temp > 0:
            if temp % 11 == 10:
                valid = False
                break
            temp //= 11
        if valid:
            S.add(x)
    
    # 2. Local Search
    # We try to swap larger elements with smaller valid ones
    improved = True
    while improved:
        improved = False
        # Sort elements in descending order to prioritize swapping large elements
        for y in sorted(list(S), reverse=True):
            S_trial = S - {y}
            # Find the smallest z not in S_trial that is valid
            best_z = None
            for z in range(1, 257):
                if z in S_trial:
                    continue
                # Check if z is valid to add to S_trial
                valid = True
                for a in S_trial:
                    s = a + z
                    if s % 2 == 0:
                        b = s // 2
                        if b in S_trial:
                            valid = False
                            break
                if valid:
                    best_z = z
                    break
            
            if best_z is not None and best_z < y:
                S = S_trial
                S.add(best_z)
                improved = True
                break
                
    return list(S)
