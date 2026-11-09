def construct():
    n = 512
    
    # Precompute all 4-term arithmetic progressions in [1, n]
    # An AP is defined by (start, d) -> (start, start+d, start+2d, start+3d)
    all_aps = []
    for d in range(1, (n - 3) // 3 + 1):
        max_start = n - 3 * d
        for start in range(1, max_start + 1):
            a = start
            b = start + d
            c = start + 2 * d
            e = start + 3 * d
            all_aps.append((a, b, c, e))
            
    # Map each element (1-indexed) to the list of APs it participates in
    elem_to_aps = [[] for _ in range(n + 1)]
    for ap in all_aps:
        for x in ap:
            elem_to_aps[x].append(ap)
            
    S = set()
    
    # Greedy construction (Stanley Sequence)
    # Add element i if it does not complete a 4-AP with existing elements
    for i in range(1, n + 1):
        valid = True
        for ap in elem_to_aps[i]:
            # Check if the other 3 elements of this AP are in S
            count = 0
            for x in ap:
                if x != i:
                    if x in S:
                        count += 1
            if count == 3:
                valid = False
                break
                
        if valid:
            S.add(i)
            
    return list(S)
