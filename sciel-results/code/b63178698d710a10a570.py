import time

def construct():
    n = 128
    start_time = time.time()
    timeout = 55.0  # Safety margin

    # Precompute 4-APs
    # ap[a][d] = (a, a+d, a+2d, a+3d)
    # We need to check constraints efficiently.
    # For each position i, store the 4-APs it belongs to.
    # And for each 4-AP, store the positions.
    
    aps = []
    pos_aps = [[] for _ in range(n+1)]
    
    for a in range(1, n+1):
        for d in range(1, (n - a) // 3 + 1):
            p1 = a
            p2 = a + d
            p3 = a + 2 * d
            p4 = a + 3 * d
            if p4 <= n:
                ap_idx = len(aps)
                aps.append((p1, p2, p3, p4))
                pos_aps[p1].append(ap_idx)
                pos_aps[p2].append(ap_idx)
                pos_aps[p3].append(ap_idx)
                pos_aps[p4].append(ap_idx)
    
    def check_ap(ap_idx, colors):
        p1, p2, p3, p4 = aps[ap_idx]
        assigned = []
        unassigned_count = 0
        for p in (p1, p2, p3, p4):
            if colors[p] != -1:
                assigned.append(colors[p])
            else:
                unassigned_count += 1
        
        if unassigned_count == 0:
            return len(set(assigned)) >= 3
        
        # Check if it's possible to assign unassigned colors
        # to satisfy the condition
        distinct_assigned = set(assigned)
        k = len(distinct_assigned)
        
        if unassigned_count == 1:
            # Need 1 color c such that |distinct_assigned U {c}| >= 3
            if k >= 3:
                return True
            if k == 2:
                # c must not be in distinct_assigned
                return True # As long as there is a color in domain not in set
                # We assume K is large enough or we check domain later?
                # For now, return True, and let the assignment loop handle it.
                # Actually, if K=2 and distinct_assigned has size 2, no color left.
                # But we are trying fixed K.
                # So we should check if there exists c in 0..K-1 not in distinct_assigned.
                # If K > k, True. Else False.
                return K > k
            if k == 1:
                # |{A, c}| >= 3 is impossible.
                return False
            if k == 0:
                return True
        
        if unassigned_count == 2:
            # Need 2 colors c1, c2 such that |distinct_assigned U {c1, c2}| >= 3
            if k >= 3:
                return True
            if k == 2:
                # Need at least one of c1, c2 to be new.
                # So we need at least 1 color in domain not in distinct_assigned.
                return K > k
            if k == 1:
                # Need |{A, c1, c2}| >= 3.
                # c1, c2 must be distinct and != A?
                # If c1=A, c2=B!=A, set is {A, B} size 2. Bad.
                # So c1, c2 must be distinct and != A.
                # We need 2 distinct colors from domain \ {A}.
                # Available colors = K - 1.
                return K - 1 >= 2
            if k == 0:
                return True
        
        if unassigned_count == 3:
            # Need 3 colors c1, c2, c3 such that |distinct_assigned U {c1, c2, c3}| >= 3
            if k >= 3:
                return True
            if k == 2:
                # Need at least one new.
                return K > k
            if k == 1:
                # Need 2 new distinct.
                return K - 1 >= 2
            if k == 0:
                return True
        
        if unassigned_count == 4:
            return True
            
        return True

    def backtrack(i, colors):
        if time.time() - start_time > timeout:
            return False
        
        if i > n:
            return True
        
        # Heuristic: Try colors 0 to K-1
        # Maybe sort by something?
        for c in range(K):
            colors[i] = c
            
            # Check all APs involving i
            valid = True
            for ap_idx in pos_aps[i]:
                if not check_ap(ap_idx, colors):
                    valid = False
                    break
            
            if valid:
                if backtrack(i+1, colors):
                    return True
            
        colors[i] = -1
        return False

    # Try increasing K
    for K in range(3, 10):
        colors = [-1] * (n + 1)
        colors[1] = 0 # Symmetry breaking
        # Check if initial state is valid
        ok = True
        for ap_idx in pos_aps[1]:
            if not check_ap(ap_idx, colors):
                ok = False
                break
        if not ok:
            continue
            
        if backtrack(2, colors):
            return colors[1:]
            
    # Fallback: Greedy
    colors = [-1] * (n + 1)
    for i in range(1, n + 1):
        for c in range(128):
            colors[i] = c
            valid = True
            for ap_idx in pos_aps[i]:
                if not check_ap(ap_idx, colors):
                    valid = False
                    break
            if valid:
                break
    return colors[1:]
