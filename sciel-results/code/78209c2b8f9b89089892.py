def _find_large_cap_set(n, restarts=10):
    if n <= 0:
        return []
    
    all_vecs = []
    for i in range(3**n):
        v = []
        x = i
        for _ in range(n):
            v.append(x % 3)
            x //= 3
        all_vecs.append(tuple(v))
        
    best_set = []
    
    for _ in range(restarts):
        random.shuffle(all_vecs)
        
        current_set = []
        forbidden = set()
        
        for v in all_vecs:
            if v in forbidden:
                continue
            current_set.append(v)
            v_arr = np.array(v, dtype=np.int8)
            for u in current_set:
                u_arr = np.array(u, dtype=np.int8)
                w1 = tuple((2 * u_arr - v_arr) % 3)
                forbidden.add(w1)
                w2 = tuple((2 * v_arr - u_arr) % 3)
                forbidden.add(w2)
                
        if len(current_set) > len(best_set):
            best_set = current_set
            
    return best_set
