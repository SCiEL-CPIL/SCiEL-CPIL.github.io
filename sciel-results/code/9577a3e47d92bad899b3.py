def construct():
    n = 256
    # 1. Build conflict graph
    # adj[i] is a set of nodes conflicting with i
    adj = [set() for _ in range(n + 1)]
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                adj[a].add(b)
                adj[b].add(a)
                
    # 2. Greedy Independent Set
    # We maintain a set of remaining nodes
    remaining = set(range(1, n + 1))
    result = []
    
    while remaining:
        # Find node with minimum degree in the induced subgraph on 'remaining'
        min_deg_node = None
        min_deg = float('inf')
        
        for v in remaining:
            # Degree is the number of neighbors also in 'remaining'
            deg = len(adj[v] & remaining)
            if deg < min_deg:
                min_deg = deg
                min_deg_node = v
        
        # Add the chosen node to the result
        result.append(min_deg_node)
        
        # Remove the chosen node and its neighbors from 'remaining'
        to_remove = adj[min_deg_node] | {min_deg_node}
        remaining -= to_remove
        
    return result
