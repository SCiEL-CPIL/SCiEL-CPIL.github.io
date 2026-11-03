import math

def construct(n):
    if n == 1:
        return "1"
    
    total_perms = math.factorial(n)
    seen_windows = set()
    
    # Start with identity permutation
    current = list(range(1, n + 1))
    seen_windows.add(tuple(current))
    
    # LRU tracking: last step each symbol was appended
    last_used = {i: -1 for i in range(1, n + 1)}
    step = 0
    
    while len(seen_windows) < total_perms:
        step += 1
        suffix = tuple(current[-(n-1):])
        
        # Try symbols in order of least recently used
        symbols_by_lru = sorted(range(1, n + 1), key=lambda s: last_used[s])
        
        found = False
        for s in symbols_by_lru:
            new_window = suffix + (s,)
            if new_window not in seen_windows:
                current.append(s)
                seen_windows.add(new_window)
                last_used[s] = step
                found = True
                break
        
        if not found:
            # Stuck - shouldn't happen for valid superperm construction
            # but break to avoid infinite loop
            break
    
    return "".join(map(str, current))
