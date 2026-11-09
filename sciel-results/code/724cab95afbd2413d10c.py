import random
import time

def construct():
    n = 20
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    num_edges_total = len(all_edges)
    
    def greedy_construct(order):
        adj = [0] * n
        edges = []
        for idx in order:
            u, v = all_edges[idx]
            # Check if adding (u, v) creates K2,3
            # Common neighbors: adj[u] & adj[v]
            # We need popcount(common) < 3
            common = adj[u] & adj[v]
            if common.bit_count() < 3:
                edges.append([u, v])
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return edges

    best_edges = []
    best_count = 0
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Initial run with a fixed order to have a valid baseline
    initial_order = list(range(num_edges_total))
    best_edges = greedy_construct(initial_order)
    best_count = len(best_edges)
    
    # Randomized restarts
    while time.time() < deadline:
        order = list(range(num_edges_total))
        random.shuffle(order)
        edges = greedy_construct(order)
        count = len(edges)
        if count > best_count:
            best_count = count
            best_edges = edges
            
    return best_edges
