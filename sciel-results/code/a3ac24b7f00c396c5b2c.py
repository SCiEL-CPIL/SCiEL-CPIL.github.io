import time

P = {'n': 15}

def construct():
    n = P['n']
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    A = list(range(7))       # vertices 0-6
    B = list(range(7, 15))   # vertices 7-14
    a_map = {v: i for i, v in enumerate(A)}  # 0->0, 1->1, ... 6->5
    b_map = {v: i for i, v in enumerate(B)}  # 7->0, 8->1, ... 14->6
    
    # Full edge space
    all_edges = set()
    for u in range(n):
        for v in range(u+1, n):
            all_edges.add((u, v))
    
    # Base: K_{7,8} (all inter edges)
    base_edges = set()
    for a in A:
        for b in B:
            base_edges.add(tuple(sorted((a, b))))
    
    # Intra edges (candidates to add)
    intra_edges = all_edges - base_edges
    
    def get_degrees(edges):
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        return deg
    
    def is_valid(edges):
        """Check no path s-u-v-t has deg[s] == deg[t] in the full graph."""
        deg = get_degrees(edges)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        for u, v in edges:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
        return True
    
    # Start with base
    current = set(base_edges)
    best = set(base_edges)
    best_count = len(best)
    
    # Try adding each intra edge greedily
    # Order: try to add edges that create the most diverse degree patterns
    # For intra edge (a_i, a_j): a_i gets +1, a_j gets +1
    # For intra edge (b_i, b_j): b_i gets +1, b_j gets +1
    
    # Strategy: add intra edges in an order that maximizes degree diversity
    # A intra edge makes two A vertices have higher degree than other A vertices
    # B intra edge makes two B vertices have higher degree than other B vertices
    
    # In K_{7,8}: A vertices have degree 8, B vertices have degree 7
    # Adding A-intra: two A vertices go to 9, rest stay at 8; B stay at 7
    # Adding B-intra: two B vertices go to 8, rest stay at 7; A stay at 8
    # So B-intra creates B vertices of degree 8 (matching A's 8) - potential conflict
    # A-intra creates A vertices of degree 9 (unique) - safer
    
    # Prioritize A-intra edges first
    a_intra = []
    b_intra = []
    for e in intra_edges:
        if e[0] < 7 and e[1] < 7:
            a_intra.append(e)
        else:
            b_intra.append(e)
    
    def try_add(edges, e):
        test = edges | {e}
        if is_valid(test):
            return test
        return None
    
    # Pass 1: Add A-intra edges
    for e in a_intra:
        if time.time() > deadline:
            break
        result = try_add(current, e)
        if result is not None:
            current = result
    
    if len(current) > best_count:
        best = set(current)
        best_count = len(best)
    
    # Pass 2: Add B-intra edges
    for e in b_intra:
        if time.time() > deadline:
            break
        result = try_add(current, e)
        if result is not None:
            current = result
    
    if len(current) > best_count:
        best = set(current)
        best_count = len(best)
    
    # Local search: try removing one edge and adding another (or two)
    # Also try different orderings via randomization
    import random
    random.seed(42)
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Shuffle candidates
        shuffled_intra = list(intra_edges)
        random.shuffle(shuffled_intra)
        
        curr = set(base_edges)
        
        for e in shuffled_intra:
            if time.time() > deadline:
                break
            result = try_add(curr, e)
            if result is not None:
                curr = result
        
        if len(curr) > best_count:
            best = set(curr)
            best_count = len(best)
    
    # Try: start with some pre-added edges, then greedily add more
    # Try all single pre-adds
    for pre in list(intra_edges)[:30]:
        if time.time() > deadline:
            break
        curr = set(base_edges)
        result = try_add(curr, pre)
        if result is None:
            continue
        curr = result
        
        for e in intra_edges:
            if e == pre:
                continue
            if time.time() > deadline:
                break
            result = try_add(curr, e)
            if result is not None:
                curr = result
        
        if len(curr) > best_count:
            best = set(curr)
            best_count = len(best)
    
    # Final: try to add any remaining edges to best
    for e in intra_edges:
        if e in best:
            continue
        if time.time() > deadline:
            break
        result = try_add(best, e)
        if result is not None:
            best = result
    
    # Build result
    result = [list(e) for e in best]
    result.sort()
    
    for i in range(len(result)):
        result[i][0] = int(result[i][0])
        result[i][1] = int(result[i][1])
    
    return result
