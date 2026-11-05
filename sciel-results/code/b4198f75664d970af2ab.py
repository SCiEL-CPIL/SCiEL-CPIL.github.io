import itertools

def has_k4(edges, n):
    # Build adjacency matrix or set for quick lookup
    adj = [[False]*n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
        
    # Check all combinations of 4 vertices
    for combo in itertools.combinations(range(n), 4):
        u, v, w, x = combo
        if adj[u][v] and adj[u][w] and adj[u][x] and \
           adj[v][w] and adj[v][x] and \
           adj[w][x]:
            return True
    return False

def has_independent_set_6(edges, n):
    # Build non-adjacency matrix (complement graph)
    adj = [[False]*n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
        
    # Independent set in G is a clique in G_complement
    # We need a clique of size 6 in G_complement
    # G_complement edge (u,v) exists if NOT adj[u][v] and u != v
    
    # Use backtracking to find clique of size 6 in complement
    # Or simply check all combinations of 6
    # n <= 40, C(40,6) is ~ 3.8 million. This might be slow in Python if done naively for multiple graphs.
    # 3.8M * 15 checks = 57M ops. Might be close to 120s limit if we try many graphs.
    # Better to use a faster clique finding or limit the candidates.
    
    # Let's use a simple recursive clique finder on the complement graph.
    comp_adj = [[False]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if not adj[i][j]:
                comp_adj[i][j] = True
                comp_adj[j][i] = True
                
    def find_clique(vertices, size):
        if size == 0:
            return True
        if len(vertices) < size:
            return False
        if size == 1:
            return True
            
        # Pick a vertex, branch on including or not
        # Simple backtracking
        v = vertices[0]
        rest = vertices[1:]
        
        # Include v: must be connected to all others in the clique
        # We need to find a clique of size-1 in the neighbors of v within rest
        neighbors = [u for u in rest if comp_adj[v][u]]
        if find_clique(neighbors, size-1):
            return True
            
        # Exclude v
        if find_clique(rest, size):
            return True
            
        return False

    return find_clique(list(range(n)), 6)

def construct():
    candidates = []
    
    # Generate Paley graphs for primes q
    # q must be 1 mod 4 for Paley graph to be undirected symmetric in the standard way?
    # Actually, Paley graph is defined for q = 1 mod 4.
    # Primes: 17, 29, 37, 41, 53...
    # Max n is 40.
    
    primes = [17, 29, 37]
    
    # Try largest first? Or smallest first to ensure we have a valid one?
    # We want to maximize n.
    # Let's try 37, 29, 17.
    
    best_graph = None
    best_n = 0
    
    for q in sorted(primes, reverse=True):
        if q > 40: continue
        squares = {x * x % q for x in range(1, q)}
        edges = []
        for i in range(q):
            for j in range(i + 1, q):
                if (j - i) % q in squares:
                    edges.append([i, j])
        
        # Check validity
        if not has_k4(edges, q) and not has_independent_set_6(edges, q):
            if q > best_n:
                best_n = q
                best_graph = {'n': q, 'edges': edges}
                
    if best_graph:
        return best_graph
    else:
        # Fallback to P17 (should have been caught above if valid)
        squares = {x * x % 17 for x in range(1, 17)}
        edges = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
        return {'n': 17, 'edges': edges}
