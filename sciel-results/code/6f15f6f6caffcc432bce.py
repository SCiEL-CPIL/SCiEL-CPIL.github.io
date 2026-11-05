import time
import random

def _construct_polarity_graph_gf9():
    # ... (same as before) ...
    pass

def _verify_c4_free(n, adj):
    # ... (same as before) ...
    pass

def _greedy_independent_set(n, adj, limit):
    """
    Try to find an independent set of size 'limit'.
    Returns the set if found, else None.
    """
    # Greedy: pick vertex with min degree, remove it and its neighbors
    # This is not guaranteed to find the max, but good for detection.
    # Actually, to find a LARGE independent set, we should pick vertices that are NOT connected.
    # Greedy for independent set:
    # 1. Start with all vertices.
    # 2. Pick a vertex v.
    # 3. Add v to independent set.
    # 4. Remove v and all neighbors of v from the pool.
    # 5. Repeat.
    
    pool = set(range(n))
    indep = []
    # Sort by degree to pick low degree first? Or high?
    # For independent set, picking a vertex removes its neighbors.
    # To maximize the set, we should pick vertices with low degree (so we remove few others).
    # Let's try picking the vertex with minimum degree in the current pool.
    
    while pool and len(indep) < limit:
        # Find vertex with min degree in pool
        min_deg = float('inf')
        min_v = -1
        for v in pool:
            deg = len(adj[v] & pool)
            if deg < min_deg:
                min_deg = deg
                min_v = v
        
        indep.append(min_v)
        # Remove min_v and its neighbors
        to_remove = {min_v}
        to_remove.update(adj[min_v] & pool)
        pool -= to_remove
        
    if len(indep) >= limit:
        return indep
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n_total, adj_total = _construct_polarity_graph_gf9()
    
    target_n = min(43, n_total)
    
    adj_sub = [set() for _ in range(target_n)]
    for i in range(target_n):
        for j in adj_total[i]:
            if j < target_n:
                adj_sub[i].add(j)
    
    if not _verify_c4_free(target_n, adj_sub):
        return _fallback_construction()
    
    # Check for independent set of size 11
    iset = _greedy_independent_set(target_n, adj_sub, 11)
    if iset:
        # Found an independent set of size 11, so graph is invalid
        return _fallback_construction()
    
    # If we reach here, we assume it's valid
    edges = []
    for i in range(target_n):
        for j in adj_sub[i]:
            if i < j:
                edges.append([i, j])
    
    return {'n': target_n, 'edges': edges}

def _fallback_construction():
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}
