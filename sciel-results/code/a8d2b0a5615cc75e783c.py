<<<<<<< SEARCH
    # Pre-compute all 4-APs in [1, N]
    # A 4-AP is (a, a+d, a+2d, a+3d)
    # Store as list of tuples of 4 integers
    all_aps = []
    ap_index = {} # map number -> list of ap indices
    
    for d in range(1, (N - 1) // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            ap = (a, a + d, a + 2 * d, a + 3 * d)
            idx = len(all_aps)
            all_aps.append(ap)
            for x in ap:
                if x not in ap_index:
                    ap_index[x] = []
                ap_index[x].append(idx)
            
    num_aps = len(all_aps)
    
    # Helper to check if adding x is valid given current in_set
    def can_add(x, in_set):
        if x not in ap_index:
            return True
        for idx in ap_index[x]:
            ap = all_aps[idx]
            # Check if the other 3 elements are in set
            count = 0
            for y in ap:
                if y != x:
                    if in_set[y]:
                        count += 1
                    else:
                        break
            if count == 3:
                return False
        return True

    # Initial set: Base-4 digits 0 and 1
    in_set = np.zeros(N + 1, dtype=bool)
    for i in range(1, N + 1):
        # Check if base 4 representation has only 0s and 1s
        x = i
        valid = True
        while x > 0:
            digit = x % 4
            if digit > 1:
                valid = False
                break
            x //= 4
        if valid:
            in_set[i] = True
=======
    # Pre-compute all 4-APs in [1, N]
    # A 4-AP is (a, a+d, a+2d, a+3d)
    ap_others = {x: [] for x in range(1, N + 1)}
    
    for d in range(1, (N - 1) // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            ap = (a, a + d, a + 2 * d, a + 3 * d)
            a, b, c, dd = ap
            ap_others[a].append((b, c, dd))
            ap_others[b].append((a, c, dd))
            ap_others[c].append((a, b, dd))
            ap_others[dd].append((a, b, c))
            
    # Helper to check if adding x is valid given current in_set
    def can_add(x, in_set):
        for o in ap_others[x]:
            if in_set[o[0]] and in_set[o[1]] and in_set[o[2]]:
                return False
        return True

    # Initial set: Base-4 digits 0 and 1
    base_in_set = np.zeros(N + 1, dtype=bool)
    for i in range(1, N + 1):
        # Check if base 4 representation has only 0s and 1s
        x = i
        valid = True
        while x > 0:
            digit = x % 4
            if digit > 1:
                valid = False
                break
            x //= 4
        if valid:
            base_in_set[i] = True
>>>>>>> REPLACE
