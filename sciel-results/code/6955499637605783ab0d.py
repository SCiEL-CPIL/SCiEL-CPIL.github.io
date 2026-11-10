def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_set = []
    best_set_set = set(best_set)
    
    orders = [
        list(range(192, 0, -1)),
        list(range(1, 193)),
    ]
    for _ in range(3):
        o = list(range(1, 193))
        random.shuffle(o)
        orders.append(o)
        
    for order in orders:
        if time.time() > deadline:
            break
        S = []
        S_set = set()
        for x in order:
            if x in S_set: continue
            if can_add(S, x):
                S.append(x)
                S_set.add(x)
        if len(S) > len(best_set):
            best_set = S[:]
            best_set_set = S_set.copy()
            
    current_set = best_set[:]
    current_set_set = best_set_set.copy()
    
    while time.time() < deadline:
        added = False
        # Try to grow
        # Iterate through all possible y
        for y in range(1, 193):
            if y in current_set_set:
                continue
            if can_add(current_set, y):
                current_set.append(y)
                current_set_set.add(y)
                added = True
                if len(current_set) > len(best_set):
                    best_set = current_set[:]
                    best_set_set = current_set_set.copy()
                break
        
        if added:
            continue
            
        # Try swap
        if not current_set:
            break
            
        x = random.choice(current_set)
        candidates = [y for y in range(1, 193) if y not in current_set_set]
        if not candidates:
            break
        y = random.choice(candidates)
        
        # Create temp set
        S_temp = [v for v in current_set if v != x]
        # S_temp_set not needed for can_add, only list
        if can_add(S_temp, y):
            current_set = S_temp
            current_set.append(y)
            current_set_set.remove(x)
            current_set_set.add(y)
            
    return best_set
