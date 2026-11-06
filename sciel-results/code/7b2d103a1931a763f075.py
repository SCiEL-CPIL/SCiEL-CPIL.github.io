    def build_greedy_graph(n, seed_val):
        random.seed(seed_val)
        adj = [set() for _ in range(n)]
        edges = []
        
        # Generate all possible edges with random weights to break ties
        possible_edges = []
        for i in range(n):
            for j in range(i+1, n):
                possible_edges.append((i, j, random.random()))
        
        # Sort by random weight to simulate different orders
        # Or use a heuristic?
        # Let's stick to random shuffle for diversity.
        random.shuffle(possible_edges)
        
        for u, v, _ in possible_edges:
            if not has_c4(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)
                edges.append([u, v])
                
        return n, adj, edges
