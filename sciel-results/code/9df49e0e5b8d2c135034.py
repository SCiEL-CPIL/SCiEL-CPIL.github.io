import time
import random

def construct():
    n = 16
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Grötzsch graph edges (11 vertices 0-10)
    # C5: 0-1-2-3-4-0
    # Mycielski layer 5-9
    # Universal 10
    grotzsch_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), # Wait, Mycielski edges are (u, v') for (u,v) in E.
        # Let's use the standard Grötzsch edge list
    ]
    
    # Correct Grötzsch Edges:
    # Vertices 0-4 form C5.
    # Vertices 5-9 are copies.
    # Vertex 10 is universal to 5-9.
    # Edges in C5: (0,1), (1,2), (2,3), (3,4), (4,0)
    # Edges from 5-9 to 0-4: (5,0), (5,1), (6,1), (6,2), (7,2), (7,3), (8,3), (8,4), (9,4), (9,0)
    # Edges from 10 to 5-9: (10,5), (10,6), (10,7), (10,8), (10,9)
    
    grotzsch_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 0), (5, 1), 
        (6, 1), (6, 2), 
        (7, 2), (7, 3), 
        (8, 3), (8, 4), 
        (9, 4), (9, 0),
        (10, 5), (10, 6), (10, 7), (10, 8), (10, 9)
    ]

    def init_adj(edges, n):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= 1 << v
            adj[v] |= 1 << u
        return adj

    def count_edges(adj):
        return sum(bin(a).count('1') for a in adj) // 2

    def is_triangle_free_add(adj, u, v):
        # Check if adding (u,v) creates a triangle
        if (adj[u] >> v) & 1:
            return False  # Already an edge
        if (adj[u] & adj[v]) != 0:
            return False  # Common neighbor exists
        return True

    best_adj = None
    best_edges = 0
    rng = random.Random(42)

    # Pre-compute all possible edges
    all_possible_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]

    num_trials = 2000
    for i in range(num_trials):
        if time.time() > deadline:
            break

        adj = init_adj(grotzsch_edges, n)
        
        # Random shuffle of edges to add
        shuffled_edges = all_possible_edges[:]
        rng.shuffle(shuffled_edges)
        
        for u, v in shuffled_edges:
            if is_triangle_free_add(adj, u, v):
                adj[u] |= 1 << v
                adj[v] |= 1 << u

        current_edges = count_edges(adj)
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = adj[:]

    # If no trial improved (shouldn't happen), fallback to core
    if best_adj is None:
        best_adj = init_adj(grotzsch_edges, n)

    # Convert best_adj to edge list
    edges_list = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                edges_list.append([u, v])
                
    return edges_list
