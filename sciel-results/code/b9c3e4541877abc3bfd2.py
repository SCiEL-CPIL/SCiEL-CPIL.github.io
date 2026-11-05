import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # 1. Initialize Grötzsch Graph (11 vertices)
    # Vertices 0-4: C5
    # Vertices 5-9: Copies
    # Vertex 10: Apex
    
    edges = []
    # C5 edges
    for i in range(5):
        edges.append([i, (i + 1) % 5])
        
    # Copy edges
    # u_i connected to neighbors of u_i
    # Neighbors of 0: 1, 4 -> 5 connected to 1, 4
    # Neighbors of 1: 0, 2 -> 6 connected to 0, 2
    # Neighbors of 2: 1, 3 -> 7 connected to 1, 3
    # Neighbors of 3: 2, 4 -> 8 connected to 2, 4
    # Neighbors of 4: 3, 0 -> 9 connected to 3, 0
    for i in range(5):
        n1 = (i + 1) % 5
        n2 = (i - 1) % 5
        edges.append([5 + i, n1])
        edges.append([5 + i, n2])
        
    # Apex edges
    for i in range(5):
        edges.append([10, 5 + i])
        
    # Current vertices: 0..10
    # Adjacency list for fast lookup
    adj = [set() for _ in range(11)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Function to find max independent set (heuristic/backtracking)
    # For small graphs, we can use a simple backtracking or greedy with backtracking.
    # Given the size (11-16), exact backtracking is feasible.
    
    def max_independent_set(adj, n):
        # Returns the set of vertices in the max independent set
        # Simple backtracking
        best = set()
        
        def backtrack(remaining, current):
            nonlocal best
            if not remaining:
                if len(current) > len(best):
                    best = set(current)
                return
            
            # Heuristic: pick vertex with highest degree in remaining?
            # Or just pick the first one.
            # To be efficient, sort by degree in remaining.
            # But n is small.
            
            v = next(iter(remaining))
            # Option 1: Include v
            remaining.remove(v)
            current.append(v)
            # Remove neighbors of v from remaining
            neighbors = adj[v]
            new_remaining = set()
            for u in remaining:
                if u not in neighbors:
                    new_remaining.add(u)
            backtrack(new_remaining, current)
            current.pop()
            
            # Option 2: Exclude v
            remaining.add(v)
            remaining.remove(v)
            backtrack(remaining, current)
            
        backtrack(set(range(n)), [])
        return best

    # Add vertices 11..15
    for new_v in range(11, 16):
        if time.time() > deadline:
            break
            
        # Find max independent set in current graph (0..new_v-1)
        n_curr = new_v
        # Note: adj list needs to be extended or we pass size
        # We maintain adj for 0..15 eventually, but for now 0..new_v-1
        # Ensure adj has size new_v
        while len(adj) < new_v:
            adj.append(set())
            
        max_is = max_independent_set(adj, n_curr)
        
        # Connect new_v to all in max_is
        for u in max_is:
            edges.append([new_v, u])
            adj[new_v].add(u)
            adj[u].add(new_v)
            
    return edges
