def _local_search(k, g, n, deadline):
    # Generate random simple k-regular graph
    graph = _generate_random_regular(n, k)
    if graph is None:
        return None, None
    
    adj = _build_adjacency(graph, n, k)
    
    while time.time() < deadline:
        # Check girth
        girth, short_cycle = _check_girth(adj, n, g)
        if girth >= g:
            return n, graph
        
        # Try to break short_cycle
        # Pick a random edge in short_cycle
        # Try to switch it
        # ...
        
    return None, None
