def construct():
    import time
    start_time = time.time()
    DEADLINE = 80  # safety margin before 90s kill

    N = 180
    
    # Precompute conflict pairs: conflict_pairs[x] = list of (u,v) where {x,u,v} is forbidden
    conflict_pairs = {i: [] for i in range(1, N + 1)}
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            diff = b - a
            num = a * b
            if num % diff != 0:
                continue
            c = num // diff
            if c > b and c <= N:
                conflict_pairs[a].append((b, c))
                conflict_pairs[b].append((a, c))
                conflict_pairs[c].append((a, b))
    
    def can_add(x, current_set):
        for (u, v) in conflict_pairs[x]:
            if u in current_set and v in current_set:
                return False
        return True
    
    # Start with [91, 180]
    result = set(range(91, N + 1))
    
    # Greedy: add from 90 down to 1
    for x in range(90, 0, -1):
        if time.time() - start_time > DEADLINE:
            break
        if can_add(x, result):
            result.add(x)
    
    # Local improvement: try to add blocked elements by removing one blocker
    improved = True
    while improved and time.time() - start_time < DEADLINE:
        improved = False
        candidates = [y for y in range(1, N + 1) if y not in result]
        
        for y in candidates:
            if time.time() - start_time > DEADLINE:
                break
            
            # Find blocking pairs
            blocking_pairs = []
            for (u, v) in conflict_pairs[y]:
                if u in result and v in result:
                    blocking_pairs.append((u, v))
            
            if not blocking_pairs:
                result.add(y)
                improved = True
                continue
            
            # Try removing one element from each blocking pair
            added = False
            for (u, v) in blocking_pairs:
                if added:
                    break
                
                # Try removing u
                if u in result:
                    result.discard(u)
                    if can_add(y, result):
                        result.add(y)
                        improved = True
                        added = True
                    else:
                        result.add(u)
                
                # Try removing v
                if v in result and not added:
                    result.discard(v)
                    if can_add(y, result):
                        result.add(y)
                        improved = True
                        added = True
                    else:
                        result.add(v)
    
    return list(result)
