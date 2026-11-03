import numpy as np
import random
import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    time_limit = 115.0
    
    # Moore bound calculation
    # For k=3, g=16: 1 + 3 * (2^(g/2) - 1)
    # General: 1 + k * sum_{i=0}^{g/2 - 1} (k-1)^i
    if g % 2 != 0:
        # Girth must be even for regular graphs? No, but for cage problems usually even.
        # If g is odd, we look for girth >= g.
        pass
        
    # Calculate lower bound n_min
    n_min = 1
    term = 1
    # g/2 is the radius
    radius = g // 2
    for i in range(radius):
        n_min += k * term
        term *= (k - 1)
        
    # Ensure n_min is even (for k=3, sum of degrees is 3n, so 3n must be even -> n even)
    if n_min % 2 != 0:
        n_min += 1
        
    # Try random search for n from n_min to 936
    # We cap at 936 because we have a known good graph there.
    max_n = 936
    
    for n in range(n_min, max_n + 1):
        if time.time() - start_time > time_limit - 5.0:
            break
            
        # Time budget for this n
        t_start = time.time()
        t_budget = 2.0 # 2 seconds per n
        
        # Try to find a graph with this n
        result = try_random_graph(n, k, g, t_budget, t_start)
        if result is not None:
            return result
            
    # Fallback to algebraic construction
    return construct_algebraic(k, g)

def try_random_graph(n, k, g, budget, start_time):
    """
    Attempts to find a k-regular graph with girth >= g on n vertices.
    Returns (n, edges) or None.
    """
    # Generate random k-regular graph
    # Use Configuration Model
    # Stubs: n * k
    # Pair them up
    
    # Optimization: We can run multiple attempts
    while time.time() - start_time < start_time + budget:
        edges = generate_random_k_regular(n, k)
        if edges is None:
            continue
            
        # Check girth
        girth = compute_girth(n, edges, k, g)
        if girth >= g:
            return n, edges
            
    return None

def generate_random_k_regular(n, k):
    """
    Generates a random k-regular graph on n vertices.
    Returns list of edges or None if failed (e.g. odd sum of degrees).
    """
    if (n * k) % 2 != 0:
        return None
        
    stubs = [i for i in range(n) for _ in range(k)]
    random.shuffle(stubs)
    edges = []
    adj = [set() for _ in range(n)]
    
    for i in range(0, n * k, 2):
        u = stubs[i]
        v = stubs[i+1]
        if u == v:
            return None
        if v in adj[u]:
            return None
        adj[u].add(v)
        adj[v].add(u)
        edges.append((min(u, v), max(u, v)))
        
    return edges

def compute_girth(n, edges, k, target_g):
    """
    Computes the girth of the graph.
    Returns the minimum cycle length.
    Stops early if girth < target_g is found?
    No, we need to know if it's >= target_g.
    So we can stop BFS at depth target_g // 2.
    If no cycle found within that depth, girth >= target_g.
    """
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    min_girth = float('inf')
    max_depth = target_g // 2
    
    for start in range(n):
        # BFS from start
        visited = [-1] * n
        parent = [-1] * n
        depth = [0] * n
        q = deque([start])
        visited[start] = start
        
        while q:
            u = q.popleft()
            if depth[u] >= max_depth:
                continue
            for v in adj[u]:
                if visited[v] == -1:
                    visited[v] = u
                    depth[v] = depth[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u] and visited[v] != u:
                    # Found a cycle
                    # Calculate length
                    # LCA logic
                    # For speed, if we just need to know if it's < target_g,
                    # and we are at depth < max_depth, any cycle found is < target_g?
                    # Not necessarily. The cycle could be long.
                    # But if we find a back edge, the cycle length is depth[u] + depth[v] - 2*depth[lca] + 1.
                    # Since depth[u], depth[v] <= max_depth, the cycle length is <= 2*max_depth = target_g.
                    # So if we find ANY cycle within depth max_depth, the girth is <= target_g.
                    # Wait, if girth is exactly target_g, we want to accept it.
                    # The condition is "no cycle shorter than g".
                    # So if we find a cycle of length L < g, we fail.
                    # If we find a cycle of length L == g, we might still pass if no shorter ones.
                    # But our BFS stops at max_depth = g//2.
                    # If we find a cycle, its length is at most g.
                    # If length < g, we return length.
                    # If length == g, we need to check if there are shorter ones.
                    # But if we find a cycle of length L, and L < g, we can return L immediately.
                    # If L == g, we continue searching?
                    # Actually, if we find a cycle of length L, and L < g, the graph is invalid.
                    # We can return L (or just 1 to indicate failure).
                    
                    # Let's compute the exact length to be sure.
                    # Trace back to LCA
                    l = u
                    r = v
                    # Since BFS is level order, one node is ancestor of the other? No.
                    # They are in the same or adjacent levels.
                    # To find LCA, trace back.
                    path_u = set()
                    curr = u
                    while curr != -1:
                        path_u.add(curr)
                        curr = parent[curr]
                        
                    curr = v
                    while curr not in path_u:
                        curr = parent[curr]
                    lca = curr
                    
                    cycle_len = depth[u] + depth[v] - 2 * depth[lca] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                    if min_girth < target_g:
                        return min_girth
                        
    return min_girth if min_girth != float('inf') else target_g + 1

def construct_algebraic(k, g):
    # Copy of the original algebraic code
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
