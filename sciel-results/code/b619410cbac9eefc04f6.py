def construct():
    # Build Groetzsch graph
    edges = []
    # Cycle 0-1-2-3-4-0
    for i in range(5):
        edges.append([i, (i+1)%5])
    # Copies 5-9
    for i in range(5):
        edges.append([5+i, 5+(i+1)%5])
        edges.append([i, 5+i])
        edges.append([i, 5+(i-1)%5])
    # Top 10
    for i in range(5):
        edges.append([10, 5+i])
        
    # Find max independent set in G2 (vertices 0-10)
    # Simple backtracking or just pick a known one
    # Known max independent set in Groetzsch: {0, 2, 5, 7, 10}? 
    # Let's just use a simple greedy or exhaustive search since n=11
    adj = [[False]*11 for _ in range(11)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
        
    def find_max_ind_set():
        best = []
        def backtrack(remaining, current):
            nonlocal best
            if len(current) > len(best):
                best = current[:]
            if not remaining:
                return
            # Pick vertex with min degree in remaining?
            # Just pick first
            v = remaining[0]
            rest = remaining[1:]
            # Include v
            new_remaining = [u for u in rest if not adj[v][u]]
            backtrack(new_remaining, current + [v])
            # Exclude v
            backtrack(rest, current)
            
        backtrack(list(range(11)), [])
        return best

    max_ind = find_max_ind_set()
    
    # Add vertices 11-15
    # Connect max_ind to 11-15
    for u in max_ind:
        for v in range(11, 16):
            edges.append([u, v])
            
    return edges
