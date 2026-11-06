import time
import random

def construct():
    n = 16
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # Grötzsch graph edges (11 vertices 0-10)
    grotzsch_edges = [
        (0,1), (1,2), (2,3), (3,4), (4,0),
        (5,1), (5,4),
        (6,0), (6,2),
        (7,1), (7,3),
        (8,2), (8,4),
        (9,0), (9,3),
        (10,5), (10,6), (10,7), (10,8), (10,9)
    ]
    
    def init_adj(edges, n):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= 1 << v
            adj[v] |= 1 << u
        return adj

    def count_edges(adj):
        return sum(bin(a).count('1') for a in adj) // 2

    def is_triangle_free(adj, u, v):
        # Check if adding (u,v) creates a triangle
        if (adj[u] >> v) & 1:
            return False  # Already an edge
        if (adj[u] & adj[v]) != 0:
            return False
        return True

    def is_3_colorable(adj, n):
        # Fast bitmask-based 3-colorability check with forward checking
        colors = [-1] * n
        # Order vertices by degree descending
        order = sorted(range(n), key=lambda x: bin(adj[x]).count('1'), reverse=True)
        pos = {v: i for i, v in enumerate(order)}
        
        def backtrack(idx):
            if idx == n:
                return True
            u = order[idx]
            # Get forbidden colors from colored neighbors
            forbidden = 0
            for v in range(n):
                if (adj[u] >> v) & 1 and colors[v] >= 0:
                    forbidden |= 1 << colors[v]
            # Try colors in order of least constrained (fewest conflicts with uncolored neighbors)
            for c in range(3):
                if forbidden & (1 << c):
                    continue
                # Check if this color causes immediate conflict
                colors[u] = c
                ok = True
                for v in range(n):
                    if (adj[u] >> v) & 1 and colors[v] == c:
                        ok = False
                        break
                if ok:
                    if backtrack(idx + 1):
                        return True
                colors[u] = -1
            return False
            
        return backtrack(0)

    def greedy_complete_fast(adj, n, rng, max_iter=10000):
        """Fast greedy completion: repeatedly add a random triangle-free edge."""
        adj = adj[:]
        for _ in range(max_iter):
            # Collect all valid (u,v) pairs
            candidates = []
            for u in range(n):
                for v in range(u+1, n):
                    if not ((adj[u] >> v) & 1) and (adj[u] & adj[v]) == 0:
                        candidates.append((u, v))
            if not candidates:
                break
            u, v = rng.choice(candidates)
            adj[u] |= 1 << v
            adj[v] |= 1 << u
        return adj

    best_adj = None
    best_edges = 0
    
    # Try multiple initial solutions with different random seeds
    rng = random.Random(0)
    for seed in range(20):
        if time.time() > deadline:
            break
        rng_s = random.Random(seed)
        adj_core = init_adj(grotzsch_edges, n)
        adj_full = greedy_complete_fast(adj_core, n, rng_s)
        edges = count_edges(adj_full)
        if not is_3_colorable(adj_full, n):
            if edges > best_edges:
                best_edges = edges
                best_adj = adj_full[:]
    
    if best_adj is None:
        # Fallback: use core only
        best_adj = init_adj(grotzsch_edges, n)
        best_edges = count_edges(best_adj)

    # Local search: remove one edge, then greedily complete
    num_trials = 50000
    for i in range(num_trials):
        if time.time() > deadline:
            break
        if i % 1000 == 0 and time.time() > deadline:
            break
            
        # Pick a random edge to remove from best_adj
        all_edges = []
        for u in range(n):
            for v in range(u+1, n):
                if (best_adj[u] >> v) & 1:
                    all_edges.append((u, v))
        
        if not all_edges:
            break
            
        u, v = rng.choice(all_edges)
        
        # Remove edge
        adj_temp = best_adj[:]
        adj_temp[u] &= ~(1 << v)
        adj_temp[v] &= ~(1 << u)
        
        # Complete greedily
        adj_new = greedy_complete_fast(adj_temp, n, rng)
        new_edges = count_edges(adj_new)
        
        # Only check 3-colorability if we improved
        if new_edges > best_edges:
            if not is_3_colorable(adj_new, n):
                best_edges = new_edges
                best_adj = adj_new[:]

    # Convert best_adj to edge list
    edges_list = []
    for u in range(n):
        for v in range(u+1, n):
            if (best_adj[u] >> v) & 1:
                edges_list.append([u, v])
                
    return edges_list
