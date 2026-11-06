def construct():
    import time
    start_time = time.time()
    deadline = start_time + 85
    
    target = 123457
    max_n = target - 1
    
    lpf = [0] * (max_n + 1)
    lpf[1] = 1
    for i in range(2, max_n + 1):
        if lpf[i] == 0:
            for j in range(i, max_n + 1, i):
                lpf[j] = i
    
    best_max = float('inf')
    best_pair = (1, target - 1)
    
    half_target = target // 2
    
    # Phase 1: Coarse search
    for a in range(1, half_target + 1, 10):
        if time.time() > deadline:
            break
        b = target - a
        cur_max = max(lpf[a], lpf[b])
        if cur_max < best_max:
            best_max = cur_max
            best_pair = (a, b)
    
    # Phase 2: Fine search in promising regions
    if best_max < float('inf'):
        # Search around the best pair found
        a0 = best_pair[0]
        search_range = 50
        for a in range(max(1, a0 - search_range), min(half_target + 1, a0 + search_range + 1)):
            if time.time() > deadline:
                break
            b = target - a
            cur_max = max(lpf[a], lpf[b])
            if cur_max < best_max:
                best_max = cur_max
                best_pair = (a, b)
    
    # Phase 3: Full search if time permits
    if time.time() < deadline - 5:
        for a in range(1, half_target + 1):
            if (a & 0xFFFF) == 0 and time.time() > deadline:
                break
            b = target - a
            cur_max = max(lpf[a], lpf[b])
            if cur_max < best_max:
                best_max = cur_max
                best_pair = (a, b)
                if best_max <= 2:
                    break
    
    return [best_pair[0], best_pair[1]]
