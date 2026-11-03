import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0 # 10s buffer
    
    # 1. Try Random Search
    # Estimate reasonable n
    if k > 2:
        exp_n = int((k-1) ** (g / 2.0))
    else:
        exp_n = g * 2
        
    moore_n = _moore_bound(k, g)
    
    # Range to try
    # Start from a bit above Moore bound, go up to 3x expected
    start_n = max(moore_n, exp_n // 2)
    end_n = max(start_n + 100, int(3 * exp_n))
    
    # Step size
    step = 50 if k == 5 else 10
    
    best_result = None
    
    for n in range(start_n, end_n, step):
        if time.time() > deadline:
            break
        # Try a few random graphs for this n
        for _ in range(10):
            if time.time() > deadline:
                break
            try:
                edges = _random_k_regular(n, k)
                if edges is None: continue
                if _check_girth(n, edges, g):
                    best_result = (n, edges)
                    return best_result
            except Exception:
                pass
                
    # 2. Fallback to LUW
    if time.time() > deadline - 5:
         # Just return LUW immediately if no time
         return _luw_construct(k, g)
        
    return _luw_construct(k, g)

def _moore_bound(k, g):
    # Moore bound for k-regular graph of girth g
    # g = 2d (bipartite) or 2d+1
    if g % 2 == 0:
        d = g // 2
        # N >= 2 * sum_{i=0}^{d-1} (k-1)^i
        return 2 * sum((k-1)**i for i in range(d))
    else:
        d = g // 2
        # N >= 1 + k * sum_{i=0}^{d-1} (k-1)^i
        return 1 + k * sum((k-1)**i for i in range(d))

def _random_k_regular(n, k):
    # Simple configuration model
    # List of stubs: [0]*k + [1]*k + ...
    stubs = []
    for i in range(n):
        stubs.extend([i]*k)
    random.shuffle(stubs)
    edges = []
    seen = set()
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i+1]
        if u == v:
            return None # Self loop
        if (min(u,v), max(u,v)) in seen:
            return None # Multi-edge
        seen.add((min(u,v), max(u,v)))
        edges.append((u, v))
    return edges

def _check_girth(n, edges, g):
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # BFS from each vertex
    for start in range(n):
        # dist[v] = distance from start
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = [start]
        
        for v in q:
            d = dist[v]
            # If d >= g//2, we don't need to explore further
            # because any cycle involving v and a node at depth >= g//2 
            # would have length > g? 
            # Actually, if we are looking for cycles < g, we only need 
            # to check connections between nodes at depth < g/2.
            if d >= g // 2:
                continue
                
            for u in adj[v]:
                if dist[u] == -1:
                    dist[u] = d + 1
                    parent[u] = v
                    q.append(u)
                else:
                    # u is visited
                    # If u is parent of v, skip (back edge)
                    if u == parent[v]:
                        continue
                    # Cycle found?
                    # The cycle length is d + 1 + dist[u]
                    # But we must ensure this is the shortest cycle?
                    # BFS guarantees dist is shortest.
                    # If dist[u] < dist[v], then the edge v-u connects level d to level dist[u].
                    # Cycle length = d + 1 + dist[u].
                    cycle_len = d + 1 + dist[u]
                    if cycle_len < g:
                        return False
    return True

def _luw_construct(k, g):
    # Original seed logic
    q = k
    d = g - 5 if (g - 5) % 2 else g - 4
    if d < 3:
        d = 3

    if q == 4:
        _add_t = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        _mul_t = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        def f_add(a, b): return a ^ b
        def f_sub(a, b): return a ^ b
        def f_mul(a, b): return _mul_t[a][b]
    else:
        def f_add(a, b): return (a + b) % q
        def f_sub(a, b): return (a - b) % q
        def f_mul(a, b): return (a * b) % q

    rules = {}
    rules[2] = (1, 1)
    rules[3] = (2, 1)
    rules[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        base = c - r
        candidates = [(1, base - 2), (base - 1, 1), (base, 1), (1, base + 1)]
        rules[c] = candidates[r]

    start_key = (0, (0,) * (d + 1))
    label = {start_key: 0}
    queue = [start_key]
    edge_list = []

    for vkey in queue:
        side, coords = vkey
        vidx = label[vkey]
        for first_val in range(q):
            ncoords = [0] * (d + 1)
            ncoords[1] = first_val
            for c in range(2, d + 1):
                a, b = rules[c]
                if side == 0:
                    ncoords[c] = f_add(coords[c], f_mul(ncoords[a], coords[b]))
                else:
                    ncoords[c] = f_sub(coords[c], f_mul(coords[a], ncoords[b]))
            nkey = (1 - side, tuple(ncoords))
            if nkey not in label:
                label[nkey] = len(queue)
                queue.append(nkey)
            if side == 0:
                edge_list.append((vidx, label[nkey]))

    return len(queue), edge_list
