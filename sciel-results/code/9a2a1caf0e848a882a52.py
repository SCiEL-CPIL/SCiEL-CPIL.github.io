"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _score_graph(n, edges):
    """Compute the maximum degree sum over all triangles in the graph."""
    # Build adjacency matrix and degree list
    adj = [[0] * n for _ in range(n)]
    deg = [0] * n
    for u, v in edges:
        adj[u][v] = 1
        adj[v][u] = 1
        deg[u] += 1
        deg[v] += 1
    
    # Find all triangles and compute max degree sum
    max_sum = 0
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u][v]:
                for w in range(v + 1, n):
                    if adj[u][w] and adj[v][w]:
                        s = deg[u] + deg[v] + deg[w]
                        if s > max_sum:
                            max_sum = s
    return max_sum

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n = 16
    target_edges = 65
    
    # Start with the known construction: K_{8,8} plus one edge inside a part
    # This gives 64 + 1 = 65 edges
    initial_edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
    initial_score = _score_graph(n, initial_edges)
    
    best_edges = list(initial_edges)
    best_score = initial_score
    
    # Local search: try to improve by swapping edges
    # An edge swap removes one edge and adds another, keeping edge count constant
    # We want to reduce the maximum triangle degree sum
    
    random.seed(42)
    
    # Precompute some structures for speed
    def make_edge_set(edges):
        return set((min(u,v), max(u,v)) for u, v in edges)
    
    best_edge_set = make_edge_set(best_edges)
    
    # Build all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    all_edges_set = set(all_edges)
    
    # Current adjacency for quick triangle detection
    adj = [[0] * n for _ in range(n)]
    deg = [0] * n
    for u, v in best_edges:
        adj[u][v] = 1
        adj[v][u] = 1
        deg[u] += 1
        deg[v] += 1
    
    # For each edge, compute the set of triangles it participates in
    def get_triangles_for_edge(adj, n, u, v):
        """Find all w such that (u,v,w) is a triangle."""
        triangles = []
        for w in range(n):
            if w != u and w != v and adj[u][w] and adj[v][w]:
                triangles.append(w)
        return triangles
    
    # Simulated annealing or simple hill climbing
    # Try random edge swaps that don't create duplicate edges
    
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        
        # Pick a random edge to remove
        e_remove = random.choice(best_edges)
        u1, v1 = e_remove
        e_remove_key = (min(u1, v1), max(u1, v1))
        
        # Pick a random edge to add (not already in the graph)
        # To make this more efficient, we can bias toward edges that might help
        # But for simplicity, let's just try random non-edges
        # We need to ensure the new edge is not already present
        attempts = 0
        while attempts < 10:
            e_add = random.choice(all_edges)
            e_add_key = (min(e_add[0], e_add[1]), max(e_add[0], e_add[1]))
            if e_add_key not in best_edge_set:
                break
            attempts += 1
        else:
            continue
        
        # Perform the swap: remove e_remove, add e_add
        u2, v2 = e_add
        
        # Update adjacency and degrees
        adj[u1][v1] = 0
        adj[v1][u1] = 0
        adj[u2][v2] = 1
        adj[v2][u2] = 1
        deg[u1] -= 1
        deg[v1] -= 1
        deg[u2] += 1
        deg[v2] += 1
        
        # Update edge set
        best_edge_set.remove(e_remove_key)
        best_edge_set.add(e_add_key)
        best_edges.remove(e_remove)
        best_edges.append(e_add)
        
        # Compute new score
        new_score = _score_graph(n, best_edges)
        
        if new_score <= best_score:
            best_score = new_score
        else:
            # Revert the swap
            adj[u1][v1] = 1
            adj[v1][u1] = 1
            adj[u2][v2] = 0
            adj[v2][u2] = 0
            deg[u1] += 1
            deg[v1] += 1
            deg[u2] -= 1
            deg[v2] -= 1
            
            best_edge_set.remove(e_add_key)
            best_edge_set.add(e_remove_key)
            best_edges.remove(e_add)
            best_edges.append(e_remove)
    
    return {'n': n, 'edges': best_edges}
# EVOLVE-BLOCK-END
