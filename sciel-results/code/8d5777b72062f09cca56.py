import random

def construct():
    n = 512
    
    # Precompute all 4-APs
    ap_list = []
    elem_to_aps = [[] for _ in range(n + 1)]
    
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            idx = len(ap_list)
            ap_list.append((a, b, c, e))
            elem_to_aps[a].append(idx)
            elem_to_aps[b].append(idx)
            elem_to_aps[c].append(idx)
            elem_to_aps[e].append(idx)
            
    num_aps = len(ap_list)
    
    def run_trial(seed):
        random.seed(seed)
        
        # 1. Greedy Hitting Set (Removal)
        in_set = [True] * (n + 1)
        active_aps = set(range(num_aps))
        
        # Calculate initial degrees
        deg = [0] * (n + 1)
        for i in range(num_aps):
            a, b, c, d = ap_list[i]
            deg[a] += 1
            deg[b] += 1
            deg[c] += 1
            deg[d] += 1
            
        while active_aps:
            # Find element with max degree among those in_set
            max_deg = -1
            candidates = []
            for i in range(1, n + 1):
                if in_set[i]:
                    if deg[i] > max_deg:
                        max_deg = deg[i]
                        candidates = [i]
                    elif deg[i] == max_deg:
                        candidates.append(i)
                        
            if not candidates or max_deg == 0:
                break
                
            x = random.choice(candidates)
            in_set[x] = False
            
            # Update degrees for all APs containing x
            for ap_idx in elem_to_aps[x]:
                if ap_idx in active_aps:
                    active_aps.remove(ap_idx)
                    a, b, c, d = ap_list[ap_idx]
                    for y in (a, b, c, d):
                        if y != x and in_set[y]:
                            deg[y] -= 1
                            
        S = set(i for i in range(1, n + 1) if in_set[i])
        
        # 2. Add-Back Phase
        removed = [i for i in range(1, n + 1) if not in_set[i]]
        random.shuffle(removed)
        
        for x in removed:
            can_add = True
            for ap_idx in elem_to_aps[x]:
                a, b, c, d = ap_list[ap_idx]
                # Check if the other 3 elements are in S
                # Optimization: just check membership
                if (a in S and b in S and c in S and d in S) and x not in S:
                    # Wait, x is not in S yet. We are checking if adding x creates a conflict.
                    # The conflict is if {x, y, z, w} is a 4-AP and y,z,w are in S.
                    # So we check if the other 3 are in S.
                    pass
                    
            # Let's rewrite the check clearly
            for ap_idx in elem_to_aps[x]:
                a, b, c, d = ap_list[ap_idx]
                others = [y for y in (a, b, c, d) if y != x]
                if all(y in S for y in others):
                    can_add = False
                    break
            if can_add:
                S.add(x)
                
        return S

    best_set = set()
    
    # Run multiple trials
    # 100 trials is safe time-wise
    for trial in range(100):
        s = run_trial(trial)
        if len(s) > len(best_set):
            best_set = s
            
    return sorted(best_set)
