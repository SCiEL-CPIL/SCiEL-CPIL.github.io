import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    N = 256
    # 1. Build adjacency bitmasks
    # adj[i] is a bitmask where bit j is 1 if i and j are incompatible (edge exists)
    adj = [0] * (N + 1)
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            # Check condition: (a*b) % (a+b) == 0
            # Equivalent to: g = gcd(a,b), k = (a+b)//g, check if k divides g
            g = 1
            # Compute gcd
            x, y = a, b
            while y:
                x, y = y, x % y
            g = x
            k = (a + b) // g
            if g % k == 0:
                # Incompatible
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
                
    # 2. Greedy initialization to get a lower bound
    # We want to find a large independent set.
    # Strategy: repeatedly pick a vertex, add to set, remove it and its neighbors.
    # To maximize set size, picking vertices with low degree is often better.
    candidates = (1 << (N + 1)) - 2  # Bits 1..256 set
    greedy_set = []
    temp_candidates = candidates
    while temp_candidates:
        # Find vertex with min degree in temp_candidates
        best_v = -1
        min_deg = N + 1
        v = 1
        while (1 << v) <= temp_candidates:
            if temp_candidates & (1 << v):
                # Count neighbors in temp_candidates
                deg = bin(temp_candidates & adj[v]).count('1')
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
            v += 1
        if best_v == -1:
            break
        greedy_set.append(best_v)
        # Remove best_v and its neighbors from temp_candidates
        temp_candidates &= ~(1 << best_v)
        temp_candidates &= ~adj[best_v]
        
    best_size = len(greedy_set)
    best_set = greedy_set[:]
    
    # 3. Branch and Bound
    # We use a recursive function
    # To avoid recursion depth issues, we might need to increase limit, but 256 is small.
    import sys
    sys.setrecursionlimit(10000)
    
    def popcount(x):
        return bin(x).count('1')
        
    def search(candidates, current_size, current_set):
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return
            
        if not candidates:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set[:]
            return
            
        # Upper bound
        ub = current_size + popcount(candidates)
        if ub <= best_size:
            return
            
        # Heuristic: pick vertex with max degree in candidates
        # This helps prune faster
        v = 1
        best_v = -1
        max_deg = -1
        # Iterate through bits set in candidates
        c = candidates
        while c:
            # Get lowest set bit
            lb = c & -c
            v = lb.bit_length() - 1
            # Count neighbors in candidates
            deg = bin(candidates & adj[v]).count('1')
            if deg > max_deg:
                max_deg = deg
                best_v = v
            c ^= lb
            if time.time() > deadline:
                return
                
        if best_v == -1:
            return
            
        # Branch 1: Include best_v
        new_candidates = candidates & ~(1 << best_v) & ~adj[best_v]
        current_set.append(best_v)
        search(new_candidates, current_size + 1, current_set)
        current_set.pop()
        
        if time.time() > deadline:
            return
            
        # Branch 2: Exclude best_v
        new_candidates = candidates & ~(1 << best_v)
        search(new_candidates, current_size, current_set)
        
    # Start search
    initial_candidates = (1 << (N + 1)) - 2
    search(initial_candidates, 0, [])
    
    # Return the best set found
    # Ensure it's a list of ints
    return sorted(best_set)
