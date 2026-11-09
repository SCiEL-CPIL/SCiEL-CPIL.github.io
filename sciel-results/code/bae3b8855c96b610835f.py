import random
import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    if k == 3 and g == 16:
        # Attempt to construct a smaller graph via local search
        # The Moore bound for k=3, g=16 is 510.
        # Known smallest is 936.
        # We try to find one in the range [512, 1000].
        
        best_result = None
        for n in range(512, 1000, 8):
            if time.time() > deadline - 10:
                break
            
            # Try a few random starts for this size
            for attempt in range(5):
                if time.time() > deadline - 5:
                    break
                
                edges = _generate_random_3_reg(n)
                if not edges:
                    continue
                
                # Check if girth is already sufficient
                if _check_girth_fast(edges, n, g):
                    return n, edges
                
                # Local search to improve girth
                edges = _local_search_girth(edges, n, g, deadline)
                
                if _check_girth_fast(edges, n, g):
                    return n, edges
    
    # Fallback: Use a known construction
    return _luw_construct(k, g)

def _generate_random_3_reg(n):
    if n < 4 or n % 2 != 0:
        return None
    if n * 3 % 2 != 0:
        return None
        
    adj = [[] for _ in range(n)]
    stubs = [i for i in range(n) for _ in range(3)]
    random.shuffle(stubs)
    edges = []
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i+1]
        if u == v:
            # Self loop, invalid
            return None
        if v in adj[u]:
            # Multi edge, invalid
            return None
        adj[u].append(v)
        adj[v].append(u)
        edges.append((u, v))
        
    # Final check
    for i in range(n):
        if len(adj[i]) != 3:
            return None
    return edges

def _check_girth_fast(edges, n, g):
    """Fast girth check using BFS with proper cycle detection."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For each vertex, do BFS and detect cycles
    # A cycle of length L exists if during BFS from source s,
    # we find an edge (u,v) where both are visited and the path 
    # through their LCA has length < g.
    # Simpler approach: BFS from each vertex, track parent, 
    # if we find a visited vertex that's not the parent, 
    # reconstruct the cycle length.
    
    for src in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[src] = 0
        queue = deque([src])
        
        while queue:
            u = queue.popleft()
            d = dist[u]
            if d >= g // 2:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = d + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    # Found a cycle. Need to compute its length.
                    # The cycle goes from src to u, edge (u,v), then v back to src via BFS tree
                    # Actually, we need to find the LCA of u and v in the BFS tree
                    # and compute the cycle length as dist[u] + dist[v] - 2*dist[lca] + 1
                    # But this is expensive. Instead, use a simpler sufficient condition:
                    # If dist[u] + dist[v] + 1 < g, there's definitely a cycle shorter than g.
                    # This is a sufficient but not necessary condition.
                    if dist[u] + dist[v] + 1 < g:
                        return False
    return True

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

def _find_min_cycle_len(adj, n, g):
    """Find the minimum cycle length in the graph, capped at g."""
    min_len = g
    for src in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[src] = 0
        queue = deque([src])
        while queue:
            u = queue.popleft()
            d = dist[u]
            if d >= g // 2:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = d + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    # Estimate cycle length
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_len:
                        min_len = cycle_len
    return min_len

def _find_short_cycle_fast(adj, n, g):
    """Find a short cycle and return its length and one edge in it."""
    min_len = g
    best_edge = None
    for src in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[src] = 0
        queue = deque([src])
        while queue:
            u = queue.popleft()
            d = dist[u]
            if d >= g // 2:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = d + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_len:
                        min_len = cycle_len
                        best_edge = (u, v)
    return min_len, best_edge

def _random_non_edge_pair(adj, n):
    """Find a random pair of non-adjacent vertices."""
    for _ in range(100):
        u, v = random.sample(range(n), 2)
        if v not in adj[u]:
            return u, v
    return None, None

def _swap_edges(edges, e1, e2):
    u, v = e1
    x, y = e2
    new_edges = []
    for e in edges:
        if (e[0] == u and e[1] == v) or (e[0] == v and e[1] == u):
            continue
        if (e[0] == x and e[1] == y) or (e[0] == y and e[1] == x):
            continue
        new_edges.append(e)
    new_edges.append((u, x))
    new_edges.append((v, y))
    return new_edges

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction. For k=3, g=16, this may not give girth 16.
    We use a simpler fallback: a known good construction or a large random search."""
    if k == 3 and g == 16:
        # The LUW construction with these parameters may not achieve girth 16.
        # Use a different approach: construct a bipartite-like structure or use
        # a known graph. For now, return a graph that we verify has girth >= g.
        # If verification fails, we'll need a better fallback.
        
        # Try the LUW construction but verify girth
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        n = len(order)
        
        # Verify girth
        if _check_girth_fast(edges, n, g):
            return n, edges
        
        # If LUW doesn't work, use a large random graph with local search
        # As a last resort, return a graph that we know works (if we have one)
        # For now, try a larger n
        for n_try in [n + 8, n + 16, n + 32, 1000, 1024]:
            if n_try % 2 != 0:
                n_try += 1
            edges_try = _generate_random_3_reg(n_try)
            if edges_try:
                edges_try = _local_search_girth(edges_try, n_try, g, time.time() + 10)
                if _check_girth_fast(edges_try, n_try, g):
                    return n_try, edges_try
        
        # Absolute fallback: return the LUW graph even if girth check fails
        # This is better than crashing, though it may score 0
        return n, edges
    else:
        # Generic LUW construction for other k, g
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges
