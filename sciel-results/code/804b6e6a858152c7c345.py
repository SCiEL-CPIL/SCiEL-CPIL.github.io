import time
import random
import sys

# EVOLVE-BLOCK-START

def _verify_girth_bfs(adj, n, g, deadline=None):
    """Verify girth >= g using non-backtracking BFS from each vertex.
    Only explores up to depth floor((g-1)/2) since cycles shorter than g
    have diameter at most floor((g-1)/2)."""
    max_depth = (g - 1) // 2
    for start in range(n):
        if deadline is not None and time.time() > deadline:
            return None  # timeout, unknown
        # Non-backtracking BFS: state is (vertex, prev_vertex)
        # dist[v] = shortest non-backtracking distance from start
        dist = {}
        queue = [(start, -1)]
        dist[(start, -1)] = 0
        head = 0
        while head < len(queue):
            u, prev = queue[head]
            head += 1
            d = dist[(u, prev)]
            if d >= max_depth:
                continue
            for v in adj[u]:
                if v == prev:
                    continue
                state = (v, u)
                if state not in dist:
                    dist[state] = d + 1
                    queue.append(state)
                else:
                    # Found a non-backtracking cycle
                    cycle_len = d + dist[state] + 1
                    if cycle_len < g:
                        return False
    return True

def _construct_luw(k, g, deadline=None):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q). Returns (n, edges) or None if invalid."""
    q = k
    # d is chosen so that girth >= d + 5 (for odd d) or d + 4 (for even d)
    # We want girth >= g
    # For odd d: girth >= d + 5, so d >= g - 5
    # For even d: girth >= d + 4, so d >= g - 4
    # We pick the smallest valid d
    d = g - 5
    if d % 2 == 0:
        d = g - 4
    if d < 3:
        d = 3
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    # Build the graph via BFS from the zero point
    # Vertices are (side, tuple of length d+1) where side in {0, 1}
    # side 0 = points, side 1 = lines
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # point p ~ line l iff l[c] - p[c] = l[a] * p[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index = {}
    order = []
    edges = []
    
    index[start] = 0
    order.append(start)
    
    for vi, vertex in enumerate(order):
        if deadline is not None and time.time() > deadline:
            return None
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    n = len(order)
    # Verify
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check degree
    for i in range(n):
        if len(adj[i]) != k:
            return None
    # Check no multi-edges
    for i in range(n):
        if len(adj[i]) != len(set(adj[i])):
            return None
    
    # Check girth
    result = _verify_girth_bfs(adj, n, g, deadline)
    if result is None:
        return None
    if not result:
        return None
    
    return n, edges

def construct(k, g):
    deadline = time.time() + 110.0  # safety margin before 120s limit
    
    # Try LUW construction
    result = _construct_luw(k, g, deadline)
    if result is not None:
        n, edges = result
        return n, edges
    
    # If LUW fails, raise an error
    raise ValueError("Could not construct a valid graph")

# EVOLVE-BLOCK-END
