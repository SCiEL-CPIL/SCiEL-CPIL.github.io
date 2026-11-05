import random

def construct():
    n = 512
    
    # Precompute 4-APs
    # ap_conflicts[x] = list of tuples (y, z, w) such that {x, y, z, w} is a 4-AP
    # Actually, for adding x, we need to check if y, z, w are already in selected.
    # So for each x, we store the other 3 elements for every 4-AP containing x.
    
    ap_conflicts = [[] for _ in range(n + 1)]
    
    # Generate all 4-APs in [1, n]
    # a, a+d, a+2d, a+3d
    for a in range(1, n + 1):
        for d in range(1, (n - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e > n:
                break
            # The AP is (a, b, c, e)
            # For a: conflicts with b, c, e
            ap_conflicts[a].append((b, c, e))
            # For b: conflicts with a, c, e
            ap_conflicts[b].append((a, c, e))
            # For c: conflicts with a, b, e
            ap_conflicts[c].append((a, b, e))
            # For e: conflicts with a, b, c
            ap_conflicts[e].append((a, b, c))
            
    def try_fill(order):
        selected = set()
        for x in order:
            # Check if x can be added
            valid = True
            for (y, z, w) in ap_conflicts[x]:
                if y in selected and z in selected and w in selected:
                    valid = False
                    break
            if valid:
                selected.add(x)
        return selected

    best_set = set()
    
    # 1. Forward Greedy
    s1 = try_fill(range(1, n + 1))
    if len(s1) > len(best_set):
        best_set = s1
        
    # 2. Backward Greedy
    s2 = try_fill(range(n, 0, -1))
    if len(s2) > len(best_set):
        best_set = s2
        
    # 3. Random Greedy (multiple trials)
    # Use fixed seeds for reproducibility and to ensure we don't waste time
    for trial in range(50):
        random.seed(trial)
        order = list(range(1, n + 1))
        random.shuffle(order)
        s3 = try_fill(order)
        if len(s3) > len(best_set):
            best_set = s3
            
    return sorted(best_set)
