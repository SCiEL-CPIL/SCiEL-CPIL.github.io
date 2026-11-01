"""Cage problem n(3,16): construct a 3-regular graph of girth >= 16."""
import time
import random
from collections import deque

# Deadline guard
_DEADLINE = 145.0  # seconds wall-clock safety margin before 150s kill

def _elapsed():
    return time.time() - _T0

def _time_up():
    return time.time() - _T0 > _DEADLINE

def construct(k, g):
    global _T0
    _T0 = time.time()
    
    # For k=3, g=16: use a known construction or search
    # The smallest known is 936 vertices. We'll try to construct a valid graph.
    # 
    # Strategy: Use the Lazebnik-Ustimenko-Woldar construction as baseline,
    # then try to improve via local search if time permits.
    #
    # Actually, let's use a more direct approach:
    # 1. Start with the LUW construction (gives a valid 3-regular graph with girth >= 16)
    # 2. Try to reduce size via local search (edge swaps that preserve 3-regularity and girth)
    
    # First, generate a valid baseline using LUW
    n, edges = _luw_graph(k, g)
    
    if _time_up():
        return n, edges
    
    # Try to improve via local search
    n, edges = _local_search(n, edges, k, g)
    
    return n, edges

def _luw_graph(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q)."""
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

def _build_adjacency(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _check_girth(n, adj, g):
    """Check if graph has girth >= g using BFS from each vertex (non-backtracking)."""
    # For 3-regular graphs, we can use a more efficient method
    # BFS with parent tracking to avoid backtracking
    for start in range(n):
        # BFS from start, tracking (node, prev)
        dist = {}
        queue = deque()
        for nb in adj[start]:
            queue.append((nb, start, 1))
            dist[(nb, start)] = 1
        while queue:
            node, prev, d = queue.popleft()
            if d >= g:
                break
            for nb in adj[node]:
                if nb == prev:
                    continue
                if (nb, node) in dist:
                    # Found a cycle of length d + 1
                    if d + 1 < g:
                        return False
                else:
                    dist[(nb, node)] = d + 1
                    if d + 1 < g:
                        queue.append((nb, node, d + 1))
    return True

def _local_search(n, edges, k, g):
    """Try to reduce graph size via local search. This is complex for cage problems.
    For now, return the baseline as local search on cage problems is very hard."""
    # Local search for cage problems is extremely challenging.
    # A 2-switch (edge swap) can change the girth.
    # Given the difficulty and time constraints, we'll just return the baseline.
    return n, edges

def _try_voltage_lift(k, g):
    """Try a voltage graph lift approach. This is experimental."""
    if k != 3 or g != 16:
        return None
    
    # Petersen graph has 10 vertices, girth 5
    # We need a lift of size m such that total vertices = 10*m
    # For girth 16, we need m large enough
    # Moore bound for (3,16) is 512, so m >= 52
    # Smallest known is 936, so m = 93.6... not integer
    # Actually 936/10 = 93.6, so it's not a Petersen lift
    
    # Let's try a different base graph
    # A 3-regular graph with girth >= 4 (no triangles)
    # The smallest such with 6 vertices is K_{3,3} (girth 4)
    # Lift of K_{3,3} by m gives 6m vertices
    # For 936 vertices: m = 156
    # This might work but finding the right voltages is hard
    
    return None

# Keep the global time tracking
_T0 = 0.0
