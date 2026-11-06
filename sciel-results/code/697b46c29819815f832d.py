import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Grötzsch graph initialization
    # Vertices 0-10
    # 0-4: Cycle
    # 5-9: Shadows
    # 10: Top
    
    n = 16
    # adj[i][j] is True if edge exists
    adj = [[False] * n for _ in range(n)]
    
    def add_edge(u, v):
        adj[u][v] = True
        adj[v][u] = True
        
    # 1. Cycle 0-1-2-3-4-0
    for i in range(5):
        add_edge(i, (i + 1) % 5)
        
    # 2. Shadows
    # Shadow of i is 5+i
    # If edge (u, v) exists, add (shadow_u, v) and (shadow_v, u)
    cycle_edges = [(i, (i + 1) % 5) for i in range(5)]
    for u, v in cycle_edges:
        su = 5 + u
        sv = 5 + v
        add_edge(su, v)
        add_edge(sv, u)
        
    # 3. Top 10 connected to all shadows
    for i in range(5):
        add_edge(10, 5 + i)
        
    # Get initial edge list
    initial_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                initial_edges.append([i, j])
                
    best_edges = initial_edges
    best_count = len(initial_edges)
    
    # Potential edges
    potential_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if not adj[i][j]:
                potential_edges.append((i, j))
                
    # Greedy Search with Restarts
    # We use a copy of the adjacency matrix for each run
    # To speed up, we can use a list of edges and a set for O(1) lookup, 
    # but for triangle check, adjacency matrix is better.
    
    # Since n=16, we can represent adjacency as a list of 16 integers (bitmasks)
    # adj_mask[i] has bit j set if (i,j) is an edge.
    
    def init_masks():
        masks = [0] * n
        for i in range(n):
            for j in range(n):
                if adj[i][j]:
                    masks[i] |= (1 << j)
        return masks
        
    base_masks = init_masks()
    
    def creates_triangle_mask(u, v, masks):
        # Common neighbors of u and v
        # neighbors_u = masks[u]
        # neighbors_v = masks[v]
        # If intersection is non-empty, triangle exists
        # Note: u and v are not neighbors of themselves, so no self-loop issues
        return (masks[u] & masks[v]) != 0
        
    num_iterations = 5000
    iter_count = 0
    
    while time.time() < deadline and iter_count < num_iterations:
        iter_count += 1
        
        # Copy masks
        masks = base_masks[:]
        current_edges = initial_edges[:]
        
        # Shuffle potentials
        shuffled = potential_edges[:]
        random.shuffle(shuffled)
        
        for u, v in shuffled:
            # If already added, skip (should not happen as we only add once)
            # Check triangle
            if not creates_triangle_mask(u, v, masks):
                # Add edge
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
                current_edges.append([u, v])
                
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges
            
    return best_edges
