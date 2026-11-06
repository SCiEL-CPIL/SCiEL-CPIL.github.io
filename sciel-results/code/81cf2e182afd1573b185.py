import time
import random
import itertools

def construct():
    n = 18
    start_time = time.time()
    deadline = start_time + 135  # 135s safety margin
    
    # Generate Paley(17) + isolated vertex
    residues = {1, 2, 4, 8, 9, 13, 15, 16}
    adj = [0] * n
    for u in range(17):
        for v in range(u + 1, 17):
            diff = (v - u) % 17
            if diff in residues:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    
    def is_triangle_free(mask):
        for u in range(n):
            if not (mask & (1 << u)):
                continue
            neighbors = adj[u] & mask
            # Check if neighbors form an independent set
            # If any edge exists between neighbors, triangle found
            # Iterate through bits in neighbors
            # Optimization: if neighbors == 0, continue
            if neighbors:
                # Check each neighbor x
                # We can do this by:
                # for x in range(n):
                #   if neighbors & (1 << x):
                #     if adj[x] & neighbors: return False
                # This is O(n) per u.
                # Total O(n^2).
                for x in range(n):
                    if neighbors & (1 << x):
                        if adj[x] & neighbors:
                            return False
        return True

    def get_max_tf_size():
        for k in range(n, 0, -1):
            for combo in itertools.combinations(range(n), k):
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                if is_triangle_free(mask):
                    return k
        return 0

    def can_add_edge(u, v):
        common = adj[u] & adj[v]
        for x in range(n):
            if common & (1 << x):
                if adj[x] & common:
                    return False
        return True

    current_score = get_max_tf_size()
    best_score = current_score
    best_adj = [a for a in adj]
    
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Local Search
    while time.time() < deadline:
        u, v = random.choice(all_edges)
        is_edge_present = bool(adj[u] & (1 << v))
        
        if not is_edge_present:
            if not can_add_edge(u, v):
                continue
        
        adj[u] ^= (1 << v)
        adj[v] ^= (1 << u)
        
        new_score = get_max_tf_size()
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = [a for a in adj]
        else:
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            
        if time.time() > deadline:
            break
            
    adj = best_adj
    
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                edges.append([u, v])
                
    return edges
