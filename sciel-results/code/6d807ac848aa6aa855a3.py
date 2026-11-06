import time

def construct():
    N = 256
    # Precompute APs
    # ap_list: list of (a,b,c)
    # aps_containing[y]: list of (a,b,c) containing y
    
    ap_list = []
    aps_containing = {i: [] for i in range(1, N+1)}
    
    for b in range(2, N):
        min_a = max(1, 2*b - N)
        for a in range(min_a, b):
            c = 2*b - a
            ap = (a, b, c)
            ap_list.append(ap)
            aps_containing[a].append(ap)
            aps_containing[b].append(ap)
            aps_containing[c].append(ap)
            
    # Helper to check if adding y to S (which is a set) creates a 3-AP
    # S is the set BEFORE adding y.
    # We need to check if any AP containing y has the other 2 elements in S.
    def can_add(S, y):
        for ap in aps_containing[y]:
            a, b, c = ap
            # Check the other two
            if y == a:
                if b in S and c in S:
                    return False
            elif y == b:
                if a in S and c in S:
                    return False
            else: # y == c
                if a in S and b in S:
                    return False
        return True
        
    # Initial Greedy
    S = set()
    for x in range(1, N + 1):
        if can_add(S, x):
            S.add(x)
            
    best_S = set(S)
    best_score = sum(1.0/x for x in S)
    
    start_time = time.time()
    timeout = 50.0
    
    improved = True
    while improved and (time.time() - start_time) < timeout:
        improved = False
        
        # Phase 1: Add
        # Try to add smallest y first
        for y in range(1, N + 1):
            if y not in S:
                if can_add(S, y):
                    S.add(y)
                    improved = True
                    break
        
        # Phase 2: Swap
        # Try to replace x in S with y not in S, y < x
        # Iterate x in S descending
        for x in sorted(S, reverse=True):
            if (time.time() - start_time) > timeout:
                break
            # Try y in 1..x-1
            # To maximize gain, try smallest y first
            for y in range(1, x):
                if y in S:
                    continue
                # Check if swap is valid
                # S' = S \ {x} U {y}
                # We need to check if S' is valid.
                # Since S was valid, S\{x} is valid.
                # We just need to check if adding y to S\{x} creates an AP.
                # So we check can_add(S \ {x}, y).
                
                # Optimization: if y is in S\{x} (it is not, y not in S), 
                # we just call can_add on the set S with x removed.
                # We don't want to create a new set every time.
                # can_add checks `b in S` etc.
                # If we pass S, it might see x. But x is being removed.
                # So we need to check: are the other two elements in S \ {x}?
                # Equivalent to: (other1 in S and other1 != x) and (other2 in S and other2 != x)
                
                valid_swap = True
                for ap in aps_containing[y]:
                    a, b, c = ap
                    if y == a:
                        if (b in S and b != x) and (c in S and c != x):
                            valid_swap = False
                            break
                    elif y == b:
                        if (a in S and a != x) and (c in S and c != x):
                            valid_swap = False
                            break
                    else: # y == c
                        if (a in S and a != x) and (b in S and b != x):
                            valid_swap = False
                            break
                
                if valid_swap:
                    S.remove(x)
                    S.add(y)
                    improved = True
                    break
            if improved:
                break
                
        current_score = sum(1.0/x for x in S)
        if current_score > best_score:
            best_score = current_score
            best_S = set(S)
            
    if not best_S:
        return [1]
    return list(best_S)
