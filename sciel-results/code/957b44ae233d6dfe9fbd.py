<<<<<<< SEARCH
    n = len(order)
    return n, edges
=======
    n = len(order)
    return n, edges

def _simple_3reg_fallback(n_target):
    """
    Generate a simple 3-regular graph with n_target vertices.
    This may not have high girth, but it's a valid 3-regular graph.
    Used as last resort.
    """
    if n_target < 6 or n_target % 2 != 0:
        # Find next valid size
        n_target += 1
        if n_target % 2 != 0:
            n_target += 1
    
    # Use a circulant-like construction
    # For 3-regular, use steps that give degree 3
    # Simple approach: use a random regular graph generator
    
    stubs = []
    for v in range(n_target):
        for _ in range(3):
            stubs.append(v)
    
    random.shuffle(stubs)
    
    adj = [[] for _ in range(n_target)]
    edge_set = set()
    
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i+1]
        
        # Avoid self-loops and duplicates
        attempts = 0
        while u == v or (min(u,v), max(u,v)) in edge_set:
            attempts += 1
            if attempts > 100:
                # Try to fix by swapping
                fixed = False
                for j in range(i+2, len(stubs), 2):
                    if stubs[j] != u and stubs[j+1] != u:
                        stubs[i+1], stubs[j+1] = stubs[j+1], stubs[i+1]
                        v = stubs[i+1]
                        if u != v and (min(u,v), max(u,v)) not in edge_set:
                            fixed = True
                            break
                if not fixed:
                    return None, None
                break
        
        if u == v:
            return None, None
        
        edge = (min(u, v), max(u, v))
        if edge in edge_set:
            return None, None
        
        adj[u].append(v)
        adj[v].append(u)
        edge_set.add(edge)
    
    # Verify all vertices have degree 3
    for v in range(n_target):
        if len(adj[v]) != 3:
            return None, None
    
    edges = list(edge_set)
    return n_target, edges
>>>>>>> REPLACE

Now update the main construct function to use the fallback:

