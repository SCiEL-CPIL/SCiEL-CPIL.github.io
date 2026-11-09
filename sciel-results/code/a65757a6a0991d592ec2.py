<<<<<<< SEARCH
def _local_search_girth(edges, n, g, deadline):
    """Improve girth via edge swaps. Returns improved edge list."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    best_min_len = _find_min_cycle_len(adj, n, g)
    
    for iteration in range(200):
        if time.time() > deadline - 2:
            break
        
        # Find a short cycle to break
        min_len, cycle_edge = _find_short_cycle_fast(adj, n, g)
        if min_len >= g:
            break
        
        if cycle_edge is None:
            break
        
        u, v = cycle_edge
        
        # Try multiple random swaps to break this cycle
        found = False
        for _ in range(15):
            if time.time() > deadline - 2:
                break
            x, y = _random_non_edge_pair(adj, n)
            if x is None:
                continue
            # Check validity of swap
            if u == x or u == y or v == x or v == y:
                continue
            if x in adj[u] or y in adj[v]:
                continue
            if x in adj[v] or y in adj[u]:
                continue
            
            # Perform swap
            new_edges = _swap_edges(edges, (u, v), (x, y))
            if new_edges is None:
                continue
            
            new_adj = [[] for _ in range(n)]
            for a, b in new_edges:
                new_adj[a].append(b)
                new_adj[b].append(a)
            
            # Check if girth improved
            new_min_len, _ = _find_short_cycle_fast(new_adj, n, g)
            if new_min_len > best_min_len:
                edges = new_edges
                adj = new_adj
                best_min_len = new_min_len
                found = True
                break
    
    return edges
=======
def _local_search_girth_nbw(edges, n, g, deadline):
    """Improve girth via edge swaps using NBW-based cycle detection. Returns improved edge list."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    best_min_len = _find_min_cycle_len_nbw(adj, n, g)
    
    for iteration in range(300):
        if time.time() > deadline - 3:
            break
        
        # Find a short cycle to break
        min_len, cycle_edge = _find_short_cycle_nbw(adj, n, g)
        if min_len >= g:
            break
        
        if cycle_edge is None:
            break
        
        u, v = cycle_edge
        
        # Try multiple random swaps to break this cycle
        found = False
        for _ in range(20):
            if time.time() > deadline - 3:
                break
            x, y = _random_non_edge_pair(adj, n)
            if x is None:
                continue
            # Check validity of swap
            if u == x or u == y or v == x or v == y:
                continue
            if x in adj[u] or y in adj[v]:
                continue
            if x in adj[v] or y in adj[u]:
                continue
            
            # Perform swap
            new_edges = _swap_edges(edges, (u, v), (x, y))
            if new_edges is None:
                continue
            
            new_adj = [[] for _ in range(n)]
            for a, b in new_edges:
                new_adj[a].append(b)
                new_adj[b].append(a)
            
            # Check if girth improved
            new_min_len, _ = _find_short_cycle_nbw(new_adj, n, g)
            if new_min_len > best_min_len:
                edges = new_edges
                adj = new_adj
                best_min_len = new_min_len
                found = True
                break
    
    return edges
>>>>>>> REPLACE

