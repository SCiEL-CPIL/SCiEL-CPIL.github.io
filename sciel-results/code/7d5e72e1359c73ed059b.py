def construct():
    start_time = time.time()
    time_limit = 55.0
    
    # State
    S = [False] * (N + 1)
    current_size = 0
    
    # Initialize randomly
    for i in range(1, N + 1):
        if random.random() < 0.5:
            S[i] = True
            current_size += 1
            
    best_S = S[:]
    best_size = current_size
    
    T = 50.0
    T_min = 0.01
    cooling = 0.9995
    lam = 100.0
    
    while time.time() - start_time < time_limit:
        i = random.randint(1, N)
        is_in = S[i]
        
        delta_viol = 0
        for (j, k) in triples_by_elem[i]:
            if is_in:
                # Removing i
                if j == i:
                    if S[k]: delta_viol -= 1
                elif k == i:
                    if S[j]: delta_viol -= 1
                else:
                    if S[j] and S[k]: delta_viol -= 1
            else:
                # Adding i
                if j == i:
                    if S[k]: delta_viol += 1
                elif k == i:
                    if S[j]: delta_viol += 1
                else:
                    if S[j] and S[k]: delta_viol += 1
        
        if is_in:
            delta_size = -1
        else:
            delta_size = 1
            
        delta_E = -delta_size + lam * delta_viol
        
        if delta_E < 0 or (T > 0 and random.random() < math.exp(-delta_E / T)):
            S[i] = not S[i]
            current_size += delta_size
            
            if current_size > best_size:
                best_size = current_size
                best_S = S[:]
        
        T *= cooling
        if T < T_min:
            T = T_min
            
    # Repair best_S
    S_set = best_S[:]
    
    # Check for violations and remove
    # To speed up, we can just iterate triples
    changed = True
    while changed:
        changed = False
        for (a, b, c) in triples:
            if S_set[a] and S_set[b] and S_set[c]:
                changed = True
                # Remove one
                # Heuristic: remove the one with the smallest value?
                # Or random?
                rem = random.choice([a, b, c])
                S_set[rem] = False
                break
                
    return [i for i in range(1, N + 1) if S_set[i]]
