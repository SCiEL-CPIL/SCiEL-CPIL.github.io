import time

def construct():
    """
    Returns a simple graph on n vertices (0..n-1) that is:
    - C5-free (no 5-cycle as a subgraph)
    - has no independent set of size 9
    
    Construction: 8 disjoint copies of K4.
    n = 32.
    alpha(G) = 8 (one vertex from each K4).
    K4 is C5-free, so the disjoint union is C5-free.
    """
    start_time = time.time()
    deadline = 150.0
    
    # Check time, though this construction is O(n^2) and very fast
    if time.time() > deadline - 1.0:
        return {'n': 32, 'edges': []} # Fallback, though unlikely to be hit

    n = 32
    edges = []
    
    # 8 disjoint K4s
    num_blocks = 8
    block_size = 4
    
    for b in range(num_blocks):
        base = b * block_size
        for i in range(block_size):
            for j in range(i + 1, block_size):
                edges.append([base + i, base + j])
                
    return {'n': n, 'edges': edges}
