def generate_random_triangle_free(n):
    adj = [0] * n
    order = list(range(n))
    random.shuffle(order)
    
    for idx in range(n):
        v = order[idx]
        prev_vertices = order[:idx]
        # Shuffling prev_vertices is not strictly necessary if we iterate all, 
        # but it makes the "greedy" choice less deterministic if we were to skip some.
        # With p=1.0, we add ALL safe edges.
        current_neighbors_mask = 0
        
        for u in prev_vertices:
            if (adj[u] & current_neighbors_mask) == 0:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_neighbors_mask |= (1 << u)
            
    edges = []
    for i in range(n):
        mask = adj[i]
        for j in range(i + 1, n):
            if (mask >> j) & 1:
                edges.append([i, j])
    return edges
