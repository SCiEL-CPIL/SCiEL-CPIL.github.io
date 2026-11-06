import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Fallback: Polarity graph for q=5 (n=31)
    # We will try to find n=39, but if not, return n=31.
    # Actually, let's try n=39, 38, 37... down to 31.
    
    def is_c4_free(edges, n):
        # Check if graph is C4-free
        # For each pair of edges (u,v) and (v,w) ? No.
        # For each pair of vertices u, v, count common neighbors.
        # If any pair has >= 2 common neighbors, C4 exists.
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        for u in range(n):
            for v in range(u+1, n):
                if len(adj[u].intersection(adj[v])) >= 2:
                    return False
        return True

    def max_independent_set_size(adj, n):
        # Branch and bound for maximum independent set
        # adj: list of sets of neighbors
        # Returns size of max independent set
        
        # Simple recursive search with bitmasks if n <= 64
        # Here n <= 43, so bitmask is perfect.
        
        # Convert adj to bitmasks
        adj_masks = [0] * n
        for u in range(n):
            mask = 0
            for v in adj[u]:
                mask |= (1 << v)
            adj_masks[u] = mask
            
        best = [0]
        
        def dfs(available, current_size, current_set_mask):
            if time.time() > deadline - 1: # Check deadline inside?
                return
            if available == 0:
                if current_size > best[0]:
                    best[0] = current_size
                return
                
            # Heuristic: pick a vertex with max degree in available subgraph
            # Or just pick the first available.
            # To speed up, pick vertex with max neighbors in available.
            verts = [i for i in range(n) if (available >> i) & 1]
            if not verts:
                return
                
            # Sort by degree in available subgraph (descending)
            # Degree of v in available = popcount(adj_masks[v] & available)
            def deg(v):
                return bin(adj_masks[v] & available).count('1')
                
            verts.sort(key=deg, reverse=True)
            v = verts[0]
            
            # Branch 1: Include v
            # Remove v and its neighbors from available
            new_available = available & ~(1 << v) & ~adj_masks[v]
            dfs(new_available, current_size + 1, current_set_mask | (1 << v))
            
            # Branch 2: Exclude v
            new_available = available & ~(1 << v)
            dfs(new_available, current_size, current_set_mask)
            
        initial_available = (1 << n) - 1
        dfs(initial_available, 0, 0)
        return best[0]

    def generate_random_c4_free(n):
        edges = []
        adj = [set() for _ in range(n)]
        
        # Random order of all pairs
        pairs = list(itertools.combinations(range(n), 2))
        random.shuffle(pairs)
        
        for u, v in pairs:
            # Check if adding (u, v) creates C4
            # Common neighbors of u and v must be empty
            if adj[u].isdisjoint(adj[v]):
                edges.append([u, v])
                adj[u].add(v)
                adj[v].add(u)
                
        return edges, adj

    # Try to find a valid graph
    best_n = 30
    best_edges = []
    
    # Generate a valid n=30 graph (10 triangles) as initial best
    for t in range(10):
        a, b, c = 3*t, 3*t+1, 3*t+2
        best_edges += [[a,b], [a,c], [b,c]]
        
    # Try n=39 down to 31
    for n in range(39, 30, -1):
        if time.time() > deadline - 5:
            break
            
        # Try multiple random seeds
        trials = 0
        while time.time() < deadline - 5:
            trials += 1
            edges, adj = generate_random_c4_free(n)
            
            # Check alpha
            alpha = max_independent_set_size(adj, n)
            
            if alpha <= 10:
                best_n = n
                best_edges = edges
                break
                
            # Optional: Local search to reduce alpha?
            # If alpha is 11, maybe we can add an edge to break the independent set?
            # But adding an edge might create C4.
            # We can try to add edges that don't create C4 and reduce alpha.
            # This is complex. Let's just restart.
            
            # If trials > 50, break and try smaller n?
            # Or keep going?
            # Random greedy is fast.
            
    return {'n': best_n, 'edges': best_edges}
