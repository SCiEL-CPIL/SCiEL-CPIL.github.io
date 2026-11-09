def construct():
    import random
    import time
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def is_triangle_free(edge_set):
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        for u, v in edge_set:
            if adj[u] & adj[v]:
                return False
        return True
    
    def is_3_colorable(edge_set):
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        # Backtracking 3-coloring
        color = [-1] * n
        
        def backtrack(v):
            if v == n:
                return True
            for c in range(3):
                if all(color[nei] != c for nei in adj[v]):
                    color[v] = c
                    if backtrack(v + 1):
                        return True
                    color[v] = -1
            return False
        
        # Order vertices by degree descending
        order = sorted(range(n), key=lambda x: -len(adj[x]))
        pos = {v: i for i, v in enumerate(order)}
        
        def backtrack2(idx):
            if idx == n:
                return True
            v = order[idx]
            for c in range(3):
                if all(color[nei] != c for nei in adj[v]):
                    color[v] = c
                    if backtrack2(idx + 1):
                        return True
                    color[v] = -1
            return False
        
        return backtrack2(0)
    
    def count_edges(edge_set):
        return len(edge_set)
    
    best_edges = 0
    best_edge_set = set()
    
    start_time = time.time()
    time_limit = 110  # Leave some margin
    
    # Start with Grötzsch graph
    groetzsch_edges = set()
    # Cycle 0-1-2-3-4-0
    for i in range(5):
        groetzsch_edges.add((i, (i+1) % 5))
    # Mycielski: vertex 5+i connected to i and (i+1)%5... 
    # Actually standard: vertex 5+w connected to u and v where (u,w) is in cycle
    for i in range(5):
        u, w = i, (i+1) % 5
        groetzsch_edges.add((u, 5+w))
        groetzsch_edges.add((w, 5+u))
    # Vertex 10 connected to all 5..9
    for i in range(5):
        groetzsch_edges.add((5+i, 10))
    
    # Verify
    if is_triangle_free(groetzsch_edges) and not is_3_colorable(groetzsch_edges):
        best_edge_set = groetzsch_edges
        best_edges = len(groetzsch_edges)
    
    # Local search: try to add edges
    def try_add_edges(edge_set):
        """Try to add edges to edge_set while maintaining triangle-freedom."""
        edge_set = set(edge_set)
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        # Try adding each possible edge
        added = True
        while added:
            added = False
            for u, v in all_edges:
                if (u, v) not in edge_set and (v, u) not in edge_set:
                    if not (adj[u] & adj[v]):
                        # Can add
                        edge_set.add((u, v))
                        adj[u].add(v)
                        adj[v].add(u)
                        added = True
        return edge_set
    
    # Greedy completion from Grötzsch
    completed = try_add_edges(groetzsch_edges)
    if is_triangle_free(completed) and not is_3_colorable(completed):
        if len(completed) > best_edges:
            best_edges = len(completed)
            best_edge_set = completed
    
    # Randomized local search
    random.seed(42)
    iterations = 0
    while time.time() - start_time < time_limit:
        iterations += 1
        # Start from a random triangle-free graph
        # Use random greedy: pick random edges, add if triangle-free
        edge_set = set()
        adj = [set() for _ in range(n)]
        random.shuffle(all_edges)
        for u, v in all_edges:
            if not (adj[u] & adj[v]):
                edge_set.add((u, v))
                adj[u].add(v)
                adj[v].add(u)
        
        # Check if non-3-colorable
        if not is_3_colorable(edge_set):
            # Try to add more edges
            completed = try_add_edges(edge_set)
            if is_triangle_free(completed) and not is_3_colorable(completed):
                if len(completed) > best_edges:
                    best_edges = len(completed)
                    best_edge_set = completed
        
        # Also try: start from current best, remove a random edge, re-complete
        if best_edge_set and random.random() < 0.5:
            edge_list = list(best_edge_set)
            remove_edge = random.choice(edge_list)
            edge_set = best_edge_set - {remove_edge}
            # Re-complete
            completed = try_add_edges(edge_set)
            if is_triangle_free(completed) and not is_3_colorable(completed):
                if len(completed) > best_edges:
                    best_edges = len(completed)
                    best_edge_set = completed
    
    # Convert to list of lists
    return [list(e) for e in best_edge_set]
