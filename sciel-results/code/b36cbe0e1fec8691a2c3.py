def _fast_delete_pass(word, n, time_limit, start_time):
    word = list(word)
    total_perms = n * _factorial(n)
    
    def rebuild(word):
        L = len(word)
        window_perm = [-1] * (L - n + 1)
        perm_count = [0] * total_perms
        for s in range(L - n + 1):
            w = tuple(word[s:s+n])
            if len(set(w)) == n:
                idx = _PERM_INDEX.get(w, -1)
                if idx >= 0:
                    window_perm[s] = idx
                    perm_count[idx] += 1
        return window_perm, perm_count

    window_perm, perm_count = rebuild(word)
    
    improved = True
    while improved:
        if time.time() - start_time > time_limit:
            break
        improved = False
        
        # Strategy 1: L->R
        for i in range(len(word)):
            if time.time() - start_time > time_limit:
                return word
            s_min = max(0, i - n + 1)
            s_max = min(i, len(word) - n)
            
            can_delete = True
            for s in range(s_min, s_max + 1):
                if window_perm[s] >= 0:
                    if perm_count[window_perm[s]] == 1:
                        can_delete = False
                        break
            
            if can_delete:
                del word[i]
                window_perm, perm_count = rebuild(word)
                improved = True
                break
        
        # If L->R didn't improve, try R->L
        if not improved:
            for i in range(len(word) - 1, -1, -1):
                if time.time() - start_time > time_limit:
                    return word
                s_min = max(0, i - n + 1)
                s_max = min(i, len(word) - n)
                
                can_delete = True
                for s in range(s_min, s_max + 1):
                    if window_perm[s] >= 0:
                        if perm_count[window_perm[s]] == 1:
                            can_delete = False
                            break
                
                if can_delete:
                    del word[i]
                    window_perm, perm_count = rebuild(word)
                    improved = True
                    break
                    
        # If still not improved, try Random
        if not improved:
            indices = list(range(len(word)))
            random.shuffle(indices)
            for i in indices:
                if time.time() - start_time > time_limit:
                    return word
                s_min = max(0, i - n + 1)
                s_max = min(i, len(word) - n)
                
                can_delete = True
                for s in range(s_min, s_max + 1):
                    if window_perm[s] >= 0:
                        if perm_count[window_perm[s]] == 1:
                            can_delete = False
                            break
                
                if can_delete:
                    del word[i]
                    window_perm, perm_count = rebuild(word)
                    improved = True
                    break

    return word
