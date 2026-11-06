import numpy as np

def construct():
    n = 16
    target_edges = 65
    
    # Helper to calculate the score (max degree sum over triangles)
    # Vectorized for speed
    def calc_score(adj):
        degs = np.sum(adj, axis=1)
        max_sum = 0
        # Iterate over edges
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    # Common neighbors
                    common = np.where((adj[u] & adj[v]) > 0)[0]
                    if len(common) > 0:
                        # Calculate sums for all w in common
                        sums = degs[u] + degs[v] + degs[common]
                        local_max = np.max(sums)
                        if local_max > max_sum:
                            max_sum = local_max
        return max_sum

    # Initialize with K_{8,8} + 1 edge
    # This is a good starting point as it's close to the extremal graph
    adj = np.zeros((n, n), dtype=np.int8)
    for u in range(8):
        for v in range(8, 16):
            adj[u, v] = 1
            adj[v, u] = 1
    # Add one edge within Part A to reach 65 edges (64 + 1)
    # We can choose this edge to minimize initial score, but any edge works for SA
    adj[0, 1] = 1
    adj[1, 0] = 1
    
    current_score = calc_score(adj)
    best_score = current_score
    best_adj = adj.copy()
    
    rng = np.random.default_rng(42)
    
    # Simulated Annealing with Edge Swaps
    T = 10.0
    cooling = 0.999
    iterations = 500000 # Can afford more with faster score calc
    
    for i in range(iterations):
        # 1. Pick a random edge to remove
        u = rng.integers(0, n)
        v = rng.integers(0, n)
        # Retry if not an edge or self-loop
        while u == v or adj[u, v] == 0:
            u = rng.integers(0, n)
            v = rng.integers(0, n)
        
        # 2. Pick a random non-edge to add
        x = rng.integers(0, n)
        y = rng.integers(0, n)
        while x == y or adj[x, y] == 1:
            x = rng.integers(0, n)
            y = rng.integers(0, n)
            
        if (u == x and v == y) or (u == y and v == x):
            continue

        # Apply move
        adj[u, v] = 0
        adj[v, u] = 0
        adj[x, y] = 1
        adj[y, x] = 1
        
        new_score = calc_score(adj)
        
        delta = new_score - current_score
        
        # Accept if better, or with probability exp(-delta/T) if worse
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = adj.copy()
        else:
            # Revert
            adj[u, v] = 1
            adj[v, u] = 1
            adj[x, y] = 0
            adj[y, x] = 0
            
        T *= cooling
        
        # Early stop if we hit the theoretical lower bound (approx 23.4, so 23 is integer min possible if it exists)
        # Actually, degree sums are integers. If we hit 23, it's very likely optimal or close.
        if best_score <= 23:
             # We can break or continue to see if 22 is possible (unlikely)
             # Let's just break to save time, 23 is the target.
             # However, to be safe, let's run a bit more to ensure it's stable?
             # No, if we found 23, it's a valid graph. The score is an integer.
             # Let's just break.
             break

    # Extract edges from best_adj
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v]:
                edges.append([u, v])
                
    return {'n': n, 'edges': edges}
