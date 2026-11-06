import time
import random

def construct():
    start_time = time.time()
    deadline = 100.0  # Safety margin before 120s limit
    
    # Fallback: Paley graph on 17 vertices
    def paley_17():
        squares = {x * x % 17 for x in range(1, 17)}
        edges = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
        return {'n': 17, 'edges': edges}

    best_result = paley_17()
    best_n = 17

    # Try to find a valid graph for n=35
    # If that fails, try 34, 33... down to 18
    # But we should prioritize 35.
    
    for target_n in range(35, 17, -1):
        if time.time() - start_time > deadline:
            break
            
        # Allocate time for this n
        # If target_n == 35, use most of the time.
        # If smaller, use less.
        time_budget = 80.0 if target_n == 35 else 10.0
        
        found = _solve_for_n(target_n, time_budget, start_time)
        if found:
            best_result = found
            best_n = target_n
            # If we found 35, we can stop or try 36? 
            # 36 is unlikely. Let's stop if we found 35 to be safe, or try 36 with remaining time.
            if target_n == 35:
                # Try 36 with remaining time?
                remaining = deadline - (time.time() - start_time)
                if remaining > 5.0:
                    found_36 = _solve_for_n(36, remaining, start_time)
                    if found_36:
                        best_result = found_36
                        best_n = 36
                break # Stop trying lower n
            
    return best_result

def _solve_for_n(n, time_budget, start_time):
    deadline = start_time + time_budget
    
    # Initial random graph
    # Use bitsets for adjacency
    adj = [0] * n
    edges = set()
    
    # Initialize with p=0.5
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                edges.add((i, j))
    
    def is_valid():
        # Check K4
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)): continue
                for k in range(j + 1, n):
                    if not (adj[i] & (1 << k)): continue
                    if not (adj[j] & (1 << k)): continue
                    for l in range(k + 1, n):
                        if (adj[i] & (1 << l)) and (adj[j] & (1 << l)) and (adj[k] & (1 << l)):
                            return False, [(i, j, k, l)]
        # Check Indep 6
        # This is slow. We only call this if K4 check passes?
        # Or we can do a fast pre-check.
        # For now, do full check.
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j): continue
                for k in range(j + 1, n):
                    if adj[i] & (1 << k): continue
                    if adj[j] & (1 << k): continue
                    for l in range(k + 1, n):
                        if adj[i] & (1 << l): continue
                        if adj[j] & (1 << l): continue
                        if adj[k] & (1 << l): continue
                        for m in range(l + 1, n):
                            if adj[i] & (1 << m): continue
                            if adj[j] & (1 << m): continue
                            if adj[k] & (1 << m): continue
                            if adj[l] & (1 << m): continue
                            for p in range(m + 1, n):
                                if (adj[i] & (1 << p)) or (adj[j] & (1 << p)) or (adj[k] & (1 << p)) or (adj[l] & (1 << p)) or (adj[m] & (1 << p)):
                                    continue
                                return False, [(i, j, k, l, m, p)]
        return True, []

    # WalkSAT Loop
    while time.time() < deadline:
        valid, violation = is_valid()
        if valid:
            # Convert adj to edge list
            edge_list = [[u, v] for u, v in sorted(edges)]
            return {'n': n, 'edges': edge_list}
        
        # Flip a random edge in the violation
        if violation:
            verts = violation[0]
            # Pick two random vertices from the violation
            u, v = random.sample(verts, 2)
            # Flip edge
            if adj[u] & (1 << v):
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                edges.discard((min(u, v), max(u, v)))
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.add((min(u, v), max(u, v)))
        
        # Random restart if stuck?
        # We can add a counter for steps without improvement, but is_valid is expensive.
        # Just keep going.
        
    return None

# Helper to ensure we don't exceed time in the main loop
# The _solve_for_n handles its own deadline.
