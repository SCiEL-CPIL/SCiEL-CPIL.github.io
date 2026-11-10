import random
import time

def construct():
    N = 20
    start_time = time.time()
    time_limit = 115  # Safety margin for 120s limit
    
    best_edges = 0
    best_graph = None
    
    # Precompute triples
    triples = []
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                triples.append((i, j, k))
                
    # Precompute all edges
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    def get_violations(adj):
        # Returns list of (triple, common_neighbors_int)
        viols = []
        for (u, v, w) in triples:
            common = adj[u] & adj[v] & adj[w]
            if common.bit_count() >= 3:
                viols.append(((u, v, w), common))
        return viols
        
    def is_valid(adj):
        # Fast check
        for (u, v, w) in triples:
            if (adj[u] & adj[v] & adj[w]).bit_count() >= 3:
                return False
        return True
        
    def remove_edges_greedy(adj):
        # Loop until valid
        # Use a counter to avoid infinite loops if something goes wrong
        max_iter = 1000
        for _ in range(max_iter):
            if time.time() - start_time > time_limit:
                break
                
            viols = get_violations(adj)
            if not viols:
                break
            
            # Count edge frequencies in violations
            edge_count = {}
            for (triple, common) in viols:
                u, v, w = triple
                # Edges are (t, c) for t in triple, c in common
                c_bits = common
                while c_bits:
                    lsb = c_bits & -c_bits
                    c = lsb.bit_length() - 1
                    c_bits ^= lsb
                    
                    for t in (u, v, w):
                        e = (min(t, c), max(t, c))
                        edge_count[e] = edge_count.get(e, 0) + 1
            
            if not edge_count:
                break
                
            # Find max count
            max_count = max(edge_count.values())
            candidates = [e for e, c in edge_count.items() if c == max_count]
            e_to_remove = random.choice(candidates)
            
            u, v = e_to_remove
            # Remove edge
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            
        return adj

    def try_add_edges(adj):
        # Try to add missing edges greedily
        # To avoid bias, shuffle the order of attempts
        improved = True
        while improved:
            improved = False
            if time.time() - start_time > time_limit:
                return adj
                
            edges_to_try = all_edges[:]
            random.shuffle(edges_to_try)
            
            for (u, v) in edges_to_try:
                if (adj[u] >> v) & 1:
                    continue # Already exists
                
                # Try adding
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
                if is_valid(adj):
                    improved = True
                else:
                    # Revert
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    
        return adj

    # Run multiple trials
    # We can run many trials because N=20 is small
    num_trials = 100
    
    for t in range(num_trials):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize graph
        # Strategy: Mix of Complete, Dense Random, and Medium Random
        r = random.random()
        if r < 0.2:
            # Complete
            adj = [0] * N
            for i in range(N):
                for j in range(i + 1, N):
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        elif r < 0.6:
            # Dense Random
            p = 0.8 + 0.2 * random.random()
            adj = [0] * N
            for i in range(N):
                for j in range(i + 1, N):
                    if random.random() < p:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
        else:
            # Medium Random
            p = 0.5 + 0.3 * random.random()
            adj = [0] * N
            for i in range(N):
                for j in range(i + 1, N):
                    if random.random() < p:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        
        # Greedy removal
        adj = remove_edges_greedy(adj)
        
        # Local improvement
        adj = try_add_edges(adj)
        
        # Count edges
        edge_count = sum(adj[i].bit_count() for i in range(N)) // 2
        
        if edge_count > best_edges:
            best_edges = edge_count
            best_graph = adj[:]
            
    # Fallback to seed if no better found (should not happen)
    if best_graph is None or best_edges <= 36:
        # Seed: K2,18
        adj_seed = [0] * N
        for i in range(2):
            for j in range(2, N):
                adj_seed[i] |= (1 << j)
                adj_seed[j] |= (1 << i)
        best_graph = adj_seed
        best_edges = 36

    # Convert to list of edges
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if (best_graph[i] >> j) & 1:
                edges.append([i, j])
                
    return edges
