def construct():
    import time
    import random
    start_time = time.time()
    DEADLINE = 85
    n = 512
    
    # Precompute all 4-APs
    ap4_list = []
    elem_to_aps = [[] for _ in range(n+1)]
    
    idx = 0
    for d in range(1, n//3 + 1):
        for a in range(1, n - 3*d + 1):
            b = a + d
            c = a + 2*d
            e = a + 3*d
            ap4_list.append((a, b, c, e))
            for el in (a, b, c, e):
                elem_to_aps[el].append(idx)
            idx += 1
    
    num_aps = len(ap4_list)
    
    in_set = [False] * (n+1)
    ap_counts = [0] * num_aps
    
    def can_add(x):
        for ap_idx in elem_to_aps[x]:
            if ap_counts[ap_idx] == 3:
                return False
        return True
    
    def add(x):
        in_set[x] = True
        for ap_idx in elem_to_aps[x]:
            ap_counts[ap_idx] += 1
    
    def remove(x):
        in_set[x] = False
        for ap_idx in elem_to_aps[x]:
            ap_counts[ap_idx] -= 1
    
    def get_size():
        return sum(1 for x in range(1, n+1) if in_set[x])
    
    def get_set():
        return [x for x in range(1, n+1) if in_set[x]]
    
    def save_state():
        return (in_set[:], ap_counts[:])
    
    def restore_state(state):
        nonlocal in_set, ap_counts
        in_set, ap_counts = state
    
    def greedy_fill():
        changed = True
        while changed:
            changed = False
            for x in range(1, n+1):
                if not in_set[x] and can_add(x):
                    add(x)
                    changed = True
    
    # Base-3 construction (3-AP-free, hence 4-AP-free)
    for x in range(1, n+1):
        temp = x
        valid = True
        while temp > 0:
            if temp % 3 == 2:
                valid = False
                break
            temp //= 3
        if valid:
            add(x)
    
    greedy_fill()
    best_set = get_set()
    best_size = len(best_set)
    
    random.seed(42)
    
    # Phase 1: Structured - try each removal once
    for x in range(1, n+1):
        if time.time() - start_time > DEADLINE - 5:
            break
        if in_set[x]:
            saved = save_state()
            remove(x)
            greedy_fill()
            if get_size() > best_size:
                best_size = get_size()
                best_set = get_set()
            else:
                restore_state(saved)
    
    # Phase 2: Random local search
    while time.time() - start_time < DEADLINE - 2:
        candidates = [x for x in range(1,n+1) if in_set[x]]
        if not candidates:
            break
        x = random.choice(candidates)
        saved = save_state()
        remove(x)
        greedy_fill()
        if get_size() > best_size:
            best_size = get_size()
            best_set = get_set()
        else:
            restore_state(saved)
    
    return sorted(best_set)
