import time

def construct():
    start_time = time.time()
    deadline = start_time + 60.0
    
    n = 256
    # Build conflict graph
    # adj[i] is a set of j such that i and j conflict
    adj = [set() for _ in range(n + 1)]
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                adj[a].add(b)
                adj[b].add(a)
    
    # Greedy MIS on the conflict graph
    # We want a set of nodes with no edges between them.
    # This is equivalent to finding a maximal independent set.
    # To maximize size, we can use a greedy heuristic.
    # Sort by degree ascending.
    
    nodes = list(range(1, n + 1))
    # We will maintain a set of available nodes
    available = set(nodes)
    selected = []
    
    # Heuristic: pick node with lowest degree in the induced subgraph of available nodes
    # To speed up, we can just sort once by global degree, or re-sort.
    # Given n=256, re-sorting is fine.
    
    while available:
        if time.time() > deadline - 5.0:
            break
            
        # Find node with min degree in available subgraph
        min_node = -1
        min_deg = float('inf')
        for u in available:
            deg = len(adj[u] & available)
            if deg < min_deg:
                min_deg = deg
                min_node = u
        
        if min_node == -1:
            break
            
        selected.append(min_node)
        # Remove min_node and its neighbors from available
        available.discard(min_node)
        for v in adj[min_node]:
            available.discard(v)
            
    return selected
