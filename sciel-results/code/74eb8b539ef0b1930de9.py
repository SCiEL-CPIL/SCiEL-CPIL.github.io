import time

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    N = 256
    
    # Precompute conflict matrix
    conflict = [[False] * (N + 1) for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if (a * b) % (a + b) == 0:
                conflict[a][b] = True
                conflict[b][a] = True
    
    # Start with all odds (known valid, size 128)
    current = set(range(1, N + 1, 2))
    best_set = set(current)
    best_size = len(best_set)
    
    evens = list(range(2, N + 1, 2))
    odds = list(range(1, N + 1, 2))
    
    def conflicts_with(x, s):
        """Return list of elements in s that conflict with x"""
        return [y for y in s if conflict[x][y]]
    
    improved = True
    iterations = 0
    while improved and time.time() < deadline and iterations < 100:
        improved = False
        iterations += 1
        
        # Phase 1: Try to add evens (with at most 1 removal)
        for e in evens:
            if e in current:
                continue
            conf = conflicts_with(e, current)
            if len(conf) == 0:
                current.add(e)
                improved = True
                if len(current) > best_size:
                    best_size = len(current)
                    best_set = set(current)
            elif len(conf) == 1:
                c = conf[0]
                current.discard(c)
                current.add(e)
                improved = True
                if len(current) > best_size:
                    best_size = len(current)
                    best_set = set(current)
        
        # Phase 2: Try swaps (remove odd, add even)
        for o in odds:
            if o not in current:
                continue
            for e in evens:
                if e in current:
                    continue
                # Check if e conflicts with (current - {o})
                rest = current - {o}
                if not any(conflict[e][y] for y in rest):
                    current.discard(o)
                    current.add(e)
                    improved = True
                    if len(current) > best_size:
                        best_size = len(current)
                        best_set = set(current)
                    break
    
    # Verify best_set
    lst = sorted(best_set)
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if (lst[i] * lst[j]) % (lst[i] + lst[j]) == 0:
                # Fallback to all odds
                return list(range(1, 257, 2))
    
    return lst
