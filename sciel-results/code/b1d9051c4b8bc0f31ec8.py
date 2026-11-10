<<<<<<< SEARCH
    # Use numpy for vectorized operations where possible
    best_adj = None
    best_edges = 0
    rng = random.Random(42)
    
    # Direct construction approach: start with known good structure
    # Try random restarts with different densities
    for restart in range(150):
        if time.time() > deadline:
            break
        
        # Random initialization with varying density
        density = 0.3 + rng.random() * 0.4
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < density:
                    adj[i] |= bit(j)
                    adj[j] |= bit(i)
        
        # Repair and optimize
        adj = local_search(adj, max_iters=150)
        
        cnt = edge_count(adj)
        if cnt > best_edges:
            best_edges = cnt
            best_adj = list(adj)
        
        # Early termination if we find a very good solution
        if best_edges > 140:
            break
=======
    # Use numpy for vectorized operations where possible
    best_adj = None
    best_edges = 0
    rng = random.Random(42)
    
    # Deterministic base construction using structured pattern
    # Start with a high-density base: connect vertices based on modular arithmetic
    # This creates a structured graph that's close to K3,3-free
    
    def construct_base():
        """Construct a high-density base graph using structured pattern."""
        adj = [0] * n
        # Connect vertices with difference 1, 2, 3, 4, 5 (mod 20)
        # This creates a circulant-like graph
        for d in range(1, 6):
            for i in range(n):
                j = (i + d) % n
                if i < j:
                    adj[i] |= bit(j)
                    adj[j] |= bit(i)
        return adj
    
    # Try deterministic base first
    base_adj = construct_base()
    base_adj = repair_aggressive(base_adj)
    base_adj = greedy_add(base_adj)
    base_cnt = edge_count(base_adj)
    
    if base_cnt > 0:
        best_adj = list(base_adj)
        best_edges = base_cnt
    
    # Try random restarts with different densities
    for restart in range(150):
        if time.time() > deadline:
            break
        
        # Random initialization with varying density
        density = 0.3 + rng.random() * 0.4
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < density:
                    adj[i] |= bit(j)
                    adj[j] |= bit(i)
        
        # Repair and optimize
        adj = local_search(adj, max_iters=150)
        
        cnt = edge_count(adj)
        if cnt > best_edges:
            best_edges = cnt
            best_adj = list(adj)
        
        # Early termination if we find a very good solution
        if best_edges > 140:
            break
>>>>>>> REPLACE

