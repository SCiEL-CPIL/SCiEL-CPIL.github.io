import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n = 16
    
    # Grötzsch graph edges
    core_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 1), (5, 4),
        (6, 2), (6, 0),
        (7, 3), (7, 1),
        (8, 4), (8, 2),
        (9, 0), (9, 3),
        (10, 5), (10, 6), (10, 7), (10, 8), (10, 9)
    ]
    
    def make_adj_from_edges(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def is_triangle_free_add(adj, u, v):
        return (adj[u] & adj[v]) == 0

    def close_graph(adj):
        # Return a new adjacency list with all possible triangle-free edges added
        # We modify a copy
        new_adj = adj[:]
        for u in range(n):
            for v in range(u + 1, n):
                if not (new_adj[u] >> v) & 1:
                    if is_triangle_free_add(new_adj, u, v):
                        new_adj[u] |= (1 << v)
                        new_adj[v] |= (1 << u)
        return new_adj

    def count_edges(adj):
        count = 0
        for u in range(n):
            count += bin(adj[u]).count('1')
        return count // 2

    def edges_from_adj(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    base_adj = make_adj_from_edges(core_edges)
    
    best_edges = []
    best_count = 0
    
    # All possible edges involving at least one vertex from 11 to 15
    # And edges within 0-10 that are not in core (to see if we can add more to the core)
    # Actually, just try all possible edges not in base_adj
    candidate_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if not (base_adj[u] >> v) & 1:
                candidate_edges.append((u, v))
                
    random.shuffle(candidate_edges)
    
    # We will perform random restarts.
    # In each restart, we start with base_adj.
    # We randomly add a subset of candidate_edges that are triangle-free.
    # Then we close the graph.
    
    num_restarts = 2000
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        current_adj = base_adj[:]
        
        # Randomly add edges from candidate_edges
        # We iterate through a shuffled list of candidate edges
        # If we can add it, we add it with some probability?
        # Or just add all possible? No, that's the closure.
        # We want to explore different "intermediate" graphs.
        
        # Let's try adding a random subset.
        # Shuffle candidate edges
        random.shuffle(candidate_edges)
        
        for u, v in candidate_edges:
            # Try to add (u,v) with 50% probability
            if random.random() < 0.5:
                if is_triangle_free_add(current_adj, u, v):
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
        
        # Now close the graph
        closed_adj = close_graph(current_adj)
        
        # Count edges
        count = count_edges(closed_adj)
        
        if count > best_count:
            best_count = count
            best_edges = edges_from_adj(closed_adj)
            
    return best_edges
