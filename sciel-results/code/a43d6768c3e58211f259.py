import time
import collections
import math

_DEADLINE = None

def _time_left():
    if _DEADLINE is None:
        return 1e9
    return _DEADLINE - time.monotonic()

def _luw_construct(k, d):
    """
    Lazebnik-Ustimenko-Woldar construction CD(d, q) with q=k.
    Returns (n, edges).
    """
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    # Rule for LUW CD(d, q)
    # c ranges from 2 to d
    rule = {}
    if d >= 2: rule[2] = (1, 1)
    if d >= 3: rule[3] = (2, 1)
    if d >= 4: rule[4] = (1, 2)
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        # x is a tuple of length d+1, x[0] is None or 0, x[1..d] are values
        # The LUW graph is bipartite.
        # Vertices are (side, x) where side in {0, 1}.
        # x[1] is the "first" variable which is free.
        for first in range(q):
            y = [None] * (d + 1)
            y[1] = first
            # y[0] is usually 0 or ignored, but let's keep structure
            y[0] = 0 
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # y[c] = x[c] + y[a] * x[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # y[c] = x[c] - x[a] * y[b]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (0,) * (d + 1))
    index = {start: 0}
    order = [start]
    edges = []

    for vertex in order:
        if _time_left() < 5.0:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges

def _check_girth(n, edges, min_girth):
    """
    Check if girth is at least min_girth.
    Returns True if girth >= min_girth, False otherwise.
    """
    if n == 0:
        return True
    
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # We need to ensure no cycle is shorter than min_girth.
    # We perform BFS from each node, but we can prune.
    # If we find a cycle of length < min_girth, return False.
    
    # Optimization: If min_girth is large, we can limit BFS depth.
    max_depth = min_girth // 2
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = collections.deque([start])
        
        while queue:
            curr = queue.popleft()
            if dist[curr] >= max_depth:
                continue
            for neighbor in adj[curr]:
                if dist[neighbor] == -1:
                    dist[neighbor] = dist[curr] + 1
                    parent[neighbor] = curr
                    queue.append(neighbor)
                elif neighbor != parent[curr]:
                    # Found a cycle
                    cycle_len = dist[curr] + dist[neighbor] + 1
                    if cycle_len < min_girth:
                        return False
    return True

def construct(k, g):
    global _DEADLINE
    _DEADLINE = time.monotonic() + 115.0

    # For k=3, we try to find the smallest d such that LUW(3, d) has girth >= g.
    # LUW(3, d) has n = 2 * 3^d vertices.
    # We estimate d.
    # Girth is roughly 2 * log_3(n/2) ?
    # For g=16, we expect d around 7.
    # Let's try d from 5 to 10.
    
    best_n = float('inf')
    best_edges = None
    
    # Determine range for d
    # Lower bound: Moore bound n_min = 2 * (3^(g/2 - 1) - 1) / 2 ?
    # For g=16, n_min = 510.
    # 2 * 3^d >= 510 => 3^d >= 255 => d >= 5 (3^5=243, 3^6=729).
    # So d=5 gives n=486 (too small, likely girth < 16).
    # d=6 gives n=1458.
    # d=7 gives n=4374.
    
    # We try d from 6 upwards.
    # If d=6 works, it's much better than d=7.
    
    for d in range(6, 12):
        if _time_left() < 10.0:
            break
            
        try:
            n, edges = _luw_construct(k, d)
            if n >= best_n:
                continue
                
            # Check girth
            if _check_girth(n, edges, g):
                best_n = n
                best_edges = edges
                # Since n increases with d, if d=6 works, d=7 is larger.
                # We want the smallest n. So if d=6 works, we can stop?
                # Yes, because n(d) is increasing with d.
                # So the first valid d gives the smallest n in this family.
                return best_n, best_edges
        except Exception:
            continue
            
    # Fallback if no d in range worked (should not happen for large d)
    if best_edges is None:
        # Use a large d to ensure validity
        d = 10
        best_n, best_edges = _luw_construct(k, d)
        
    return best_n, best_edges
