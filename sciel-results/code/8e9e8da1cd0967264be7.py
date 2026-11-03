"""Construct a 3-regular graph with girth at least g on as few vertices as possible."""
import numpy as np
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Uses a deterministic cycle-avoiding greedy algorithm.
    """
    if k != 3:
        return _construct_general(k, g)
    
    # For k=3, g=16: use the non-backtracking walk tracking approach
    return _greedy_construct(k, g)

def _greedy_construct(k, g):
    """
    Deterministic greedy construction using non-backtracking walk tracking.
    """
    # Moore bound for (3,16): 510, n must be even so start at 512
    moore_bound = 2 * sum((k - 1) ** i for i in range(g // 2))
    start_n = moore_bound if moore_bound % 2 == 0 else moore_bound + 1
    
    # Try increasing n values
    for n in range(start_n, start_n + 50, 2):
        result = _try_greedy_n(n, k, g)
        if result is not None:
            return result
    
    # Fallback
    return _construct_fallback(k, g)

def _try_greedy_n(n, k, g):
    """
    Try to construct a k-regular graph on n vertices with girth >= g.
    """
    import random
    rng = random.Random(12345 + n)
    
    adj = [[] for _ in range(n)]
    edges = []
    
    # Precompute reachability via non-backtracking walks for cycle checking
    # We'll use a bounded BFS approach
    
    def nonbt_reach(u, v, adj, max_dist):
        """Check if v is reachable from u via a non-backtracking walk of length 1..max_dist."""
        if u == v:
            return 1  # Direct edge would be length 1, but we check separately
        visited = set()
        queue = deque()
        # State: (node, prev, dist)
        queue.append((u, -1, 0))
        visited.add((u, -1))
        
        while queue:
            node, prev, dist = queue.popleft()
            if dist >= max_dist:
                continue
            for nb in adj[node]:
                if nb == prev:
                    continue
                if nb == v:
                    return dist + 1
                state = (nb, node)
                if state not in visited:
                    visited.add(state)
                    queue.append((nb, node, dist + 1))
        return -1
    
    # Generate edges in a deterministic but shuffled order
    # To encourage good degree distribution, we'll use a multi-pass approach
    
    # Pass 1: Try to create a roughly regular graph
    # We'll iterate through vertices and try to connect them to other low-degree vertices
    
    # Create a priority queue of (degree, vertex) to process low-degree vertices first
    # But for simplicity, let's just shuffle edges and process them
    
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(all_edges)
    
    for u, v in all_edges:
        if len(adj[u]) >= k or len(adj[v]) >= k:
            continue
        if v in adj[u]:
            continue
        
        # Check if adding (u,v) creates a cycle of length < g
        # This happens if there's a non-backtracking path from u to v of length < g-1
        # (the path plus the new edge forms a cycle)
        path_len = nonbt_reach(u, v, adj, g - 2)
        if path_len != -1:
            # A path of length path_len exists, so cycle length = path_len + 1
            # We need cycle length >= g, so path_len + 1 >= g, i.e., path_len >= g - 1
            # But we searched up to g-2, so if we found anything, it's a problem
            continue
        
        adj[u].append(v)
        adj[v].append(u)
        edges.append((u, v))
    
    # Check regularity
    for i in range(n):
        if len(adj[i]) != k:
            return None
    
    # Verify girth >= g
    for start in range(n):
        visited = set()
        queue = deque()
        queue.append((start, -1, 0))
        visited.add((start, -1))
        
        while queue:
            node, prev, dist = queue.popleft()
            if dist >= g - 1:
                continue
            for nb in adj[node]:
                if nb == prev:
                    continue
                if nb == start and dist + 1 < g:
                    return None
                state = (nb, node)
                if state not in visited:
                    visited.add(state)
                    queue.append((nb, node, dist + 1))
    
    return n, edges

def _construct_fallback(k, g):
    """Fallback construction using Lazebnik-Ustimenko-Woldar graphs."""
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

def _construct_general(k, g):
    """General construction for other values of k and g."""
    return _construct_fallback(k, g)
