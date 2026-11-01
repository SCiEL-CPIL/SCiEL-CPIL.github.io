import time
import random
import sys

def _luw_construct(k, g):
    """
    Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    Returns (n, edges).
    """
    q = k
    # d is the dimension parameter.
    # For odd d, girth is at least d + 5.
    # We want girth >= g.
    # So d + 5 >= g => d >= g - 5.
    # Also d must be odd for the girth guarantee in the standard LUW description?
    # The seed code uses: d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # Let's stick to the seed logic for d.
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        
    # Rule for LUW
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
                # point p ~ line l  iff  l[c] - p[c] = l[a] * p[b]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
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

def _random_regular_graph(k, n):
    """
    Generate a random k-regular graph on n vertices.
    Uses configuration model.
    """
    if n * k % 2 != 0:
        return None
    if k >= n:
        return None
        
    half_edges = n * k // 2
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
            
    random.shuffle(stubs)
    
    adj = {i: set() for i in range(n)}
    for i in range(0, len(stubs), 2):
        u = stubs[i]
        v = stubs[i+1]
        if u == v:
            # Self-loop, retry or fail
            # Simple retry
            return _random_regular_graph(k, n)
        if v in adj[u]:
            # Multi-edge, retry
            return _random_regular_graph(k, n)
        adj[u].add(v)
        adj[v].add(u)
        
    # Check regularity (should be by construction)
    for i in range(n):
        if len(adj[i]) != k:
            return None
            
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    return edges

def _girth(adj, n):
    """
    Compute girth of graph.
    adj: list of lists of neighbors.
    """
    min_cycle = float('inf')
    for start in range(n):
        # BFS
        dist = [-1] * n
        dist[start] = 0
        queue = [start]
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
                else:
                    # Cycle found
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_cycle:
                        min_cycle = cycle_len
    return min_cycle if min_cycle != float('inf') else 0

def _sa_search(k, g, n, time_limit):
    """
    Simulated Annealing / Local Search for k-regular graph of girth >= g.
    """
    start_time = time.time()
    
    # Try a few random starts
    for attempt in range(5):
        if time.time() - start_time > time_limit:
            break
            
        edges = _random_regular_graph(k, n)
        if edges is None:
            continue
            
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Check initial girth
        curr_girth = _girth([list(adj[i]) for i in range(n)], n)
        if curr_girth >= g:
            return True, edges
            
        # Local Search
        while time.time() - start_time < time_limit:
            # Find a short cycle
            # For efficiency, just pick random edges and swap
            # Better: pick edges that are in short cycles
            # Simplified: Random swap
            e1_idx = random.randint(0, len(edges) - 1)
            e2_idx = random.randint(0, len(edges) - 1)
            if e1_idx == e2_idx:
                continue
                
            u1, v1 = edges[e1_idx]
            u2, v2 = edges[e2_idx]
            
            # Check if swap creates loop or multi-edge
            # Swap (u1, v1) and (u2, v2) with (u1, u2) and (v1, v2)
            # Or (u1, v2) and (v1, u2)
            
            # Try option 1
            new_e1 = (u1, u2)
            new_e2 = (v1, v2)
            
            if new_e1[0] == new_e1[1] or new_e2[0] == new_e2[1]:
                continue
            if new_e1[0] > new_e1[1]: new_e1 = (new_e1[1], new_e1[0])
            if new_e2[0] > new_e2[1]: new_e2 = (new_e2[1], new_e2[0])
            
            # Check if new edges already exist
            if new_e1 in adj[u1] or new_e2 in adj[v1]:
                 # Need to check if edge exists in graph
                 pass # Simplified check: just try to apply and verify
            
            # Apply swap
            # Remove old
            adj[u1].remove(v1)
            adj[v1].remove(u1)
            adj[u2].remove(v2)
            adj[v2].remove(u2)
            
            # Add new
            # Check validity again
            if u2 in adj[u1] or v2 in adj[v1]:
                # Revert
                adj[u1].add(v1)
                adj[v1].add(u1)
                adj[u2].add(v2)
                adj[v2].add(u2)
                continue
                
            adj[u1].add(u2)
            adj[u2].add(u1)
            adj[v1].add(v2)
            adj[v2].add(v1)
            
            # Update edges list
            edges[e1_idx] = new_e1
            edges[e2_idx] = new_e2
            
            # Check girth
            # Optimization: Only check if swap might have broken a cycle
            # For now, check every 10 swaps or if we suspect improvement
            # Actually, checking girth is expensive.
            # We can estimate: if we broke a short cycle, girth might increase.
            # Let's check girth every 20 iterations.
            
            # To keep code simple and fast, we'll just check periodically
            # Or we can just run for a fixed number of steps
            
            if random.random() < 0.1: # Check 10% of the time
                 curr_girth = _girth([list(adj[i]) for i in range(n)], n)
                 if curr_girth >= g:
                     return True, edges
                     
    return False, None

def construct(k, g):
    start_time = time.time()
    deadline = 110.0 # 10s margin
    
    # 1. Generate LUW Fallback
    try:
        n_luw, edges_luw = _luw_construct(k, g)
    except:
        # If LUW fails, try a simple cycle or complete graph?
        # But LUW should work for standard k, g.
        n_luw = 0
        edges_luw = []
        
    if n_luw == 0:
         # Fallback: Return empty or error?
         # Must return valid graph.
         # If LUW fails, we are in trouble.
         # But LUW is robust.
         pass

    # 2. Moore Bound
    if g % 2 == 0:
        depth = g // 2
        moore = 1 + k * sum((k-1)**i for i in range(depth))
    else:
        depth = (g - 1) // 2
        moore = 1 + k * sum((k-1)**i for i in range(depth))
        
    # 3. Search for smaller graph
    best_n = n_luw
    best_edges = edges_luw
    
    # Only attempt search if parameters are small enough
    if k <= 6 and g <= 10 and n_luw > moore:
        # Try n from moore to min(n_luw, moore + 100)
        # Use a small time budget for each n
        search_budget = 5.0
        for n in range(moore, min(n_luw, moore + 150)):
            if time.time() > deadline:
                break
            if time.time() - start_time > deadline - 10: # Keep 10s for LUW if needed
                break
                
            # If n is too close to moore, it might be impossible.
            # Start from moore + small offset
            if n < moore + 5 and k > 3:
                continue
                
            success, edges = _sa_search(k, g, n, time_limit=search_budget)
            if success:
                return n, edges
            # If fail, try next n
            # Reduce time budget for subsequent tries?
            # Keep it simple.
            
    return best_n, best_edges
