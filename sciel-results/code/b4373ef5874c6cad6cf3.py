def construct(n):
    if n == 1:
        return "1"
    
    total_perms = 1
    for i in range(2, n + 1):
        total_perms *= i
    
    word = list(range(1, n + 1))
    seen = {tuple(word)}
    
    # LRU: track when each symbol was last appended
    last_appended = {i: i for i in range(1, n + 1)}
    step = n  # current step number
    
    while len(seen) < total_perms:
        # Current window is last n-1 symbols
        window = tuple(word[-(n-1):])
        
        # Try symbols in LRU order (least recently used first)
        candidates = sorted(range(1, n + 1), key=lambda s: last_appended[s])
        
        found = False
        for s in candidates:
            new_window = window + (s,)
            # Check if it's a valid permutation (all distinct)
            if len(set(new_window)) == n:
                if new_window not in seen:
                    word.append(s)
                    seen.add(new_window)
                    last_appended[s] = step
                    step += 1
                    found = True
                    break
        
        if not found:
            # Fallback: try any symbol that creates a valid permutation window
            # Even if seen, to avoid getting stuck
            for s in range(1, n + 1):
                new_window = window + (s,)
                if len(set(new_window)) == n:
                    word.append(s)
                    seen.add(new_window)
                    last_appended[s] = step
                    step += 1
                    found = True
                    break
            
            if not found:
                # Shouldn't happen, but break to avoid infinite loop
                break
    
    return "".join(map(str, word))
