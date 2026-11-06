import random
import time

def construct():
    n = 16
    
    # Helper to check if graph is 3-colorable
    def is_3_colorable(adj):
        colors = [-1] * n
        # Backtracking search
        def backtrack(v):
            if v == n:
                return True
            for c in range(3):
                if all(colors[u] != c for u in adj[v]):
                    colors[v] = c
                    if backtrack(v + 1):
                        return True
                    colors[v] = -1
            return False
        
        # Order vertices by degree descending for better pruning
        order = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
        # Map new index to old index
        # Actually, simple backtracking on 16 vertices is fast enough if we prune well.
        # But we call this many times.
        # Let's use a simpler check or cache?
        # For SA, we need it to be fast.
        
        # Simple greedy coloring check first? No, must be exact.
        # 16 vertices is small.
        return backtrack(0)

    # Helper to check if graph is triangle-free
    def is_triangle_free(adj):
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    for w in adj[v]:
                        if w > v:
                            if w in adj[u]:
                                return False
        return True

    # Initial state: Random sparse graph
    random.seed(42)
    edges = set()
    adj = {i: set() for i in range(n)}
    
    # Start with a known 4-chromatic triangle-free graph to ensure validity initially
    # Use Grötzsch graph (11 vertices) + 5 isolated vertices
    # Or just random and hope SA finds a valid one?
    # Better to start valid.
    
    # Grötzsch edges
    groet_edges = [
        [0,1], [1,2], [2,3], [3,4], [4,0],
        [0,5], [1,6], [2,7], [3,8], [4,9],
        [5,6], [6,7], [7,8], [8,9], [9,5],
        [5,10], [6,10], [7,10], [8,10], [9,10]
    ]
    # Note: Standard Mycielski of C5.
    # Vertices 0-4 (C5), 5-9 (copies), 10 (root).
    # Edges:
    # C5: (0,1),(1,2),(2,3),(3,4),(4,0)
    # Copies: (0,5),(1,6),(2,7),(3,8),(4,9)
    # Copy-Copy: (5,6),(6,7),(7,8),(8,9),(9,5)
    # Root: (5,10),(6,10),(7,10),(8,10),(9,10)
    
    for u, v in groet_edges:
        edges.add(frozenset((u, v)))
        adj[u].add(v)
        adj[v].add(u)
        
    # Vertices 11-15 are isolated initially.
    
    def get_score(adj):
        if not is_triangle_free(adj):
            return -1000 - len(edges)
        if is_3_colorable(adj):
            return -500 - len(edges)
        return len(edges)
    
    current_score = get_score(adj)
    
    T = 100.0
    T_min = 0.1
    alpha = 0.99
    
    start_time = time.time()
    while time.time() - start_time < 100: # Keep 20s for final checks
        if T < T_min:
            break
        T *= alpha
        
        # Pick a random edge to add or remove
        if random.random() < 0.5:
            # Add edge
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
            e = frozenset((u, v))
            if e in edges:
                continue
            # Check triangle
            if adj[u] & adj[v]:
                continue
            # Apply
            edges.add(e)
            adj[u].add(v)
            adj[v].add(u)
            new_score = get_score(adj)
            delta = new_score - current_score
            if delta > 0 or random.random() < pow(2.718, delta / T):
                current_score = new_score
            else:
                # Revert
                edges.remove(e)
                adj[u].remove(v)
                adj[v].remove(u)
        else:
            # Remove edge
            if not edges:
                continue
            e = random.choice(list(edges))
            u, v = tuple(e)
            # Apply
            edges.remove(e)
            adj[u].remove(v)
            adj[v].remove(u)
            new_score = get_score(adj)
            delta = new_score - current_score
            if delta > 0 or random.random() < pow(2.718, delta / T):
                current_score = new_score
            else:
                # Revert
                edges.add(e)
                adj[u].add(v)
                adj[v].add(u)
                
    # Final verification and cleanup
    if not is_triangle_free(adj) or is_3_colorable(adj):
        # Fallback to initial valid state
        edges = set(frozenset((u, v)) for u, v in groet_edges)
        adj = {i: set() for i in range(n)}
        for u, v in groet_edges:
            adj[u].add(v)
            adj[v].add(u)
            
    return [list(e) for e in edges]
