import time
import random
import sys
from collections import deque

def _has_girth_at_least(adj, n, target_girth):
    """Check if girth >= target_girth. Returns False early if a shorter cycle is found."""
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            v = queue.popleft()
            for u in adj[v]:
                if dist[u] == -1:
                    dist[u] = dist[v] + 1
                    parent[u] = v
                    queue.append(u)
                elif u != parent[v] and v != parent[u]:
                    cycle_len = dist[v] + dist[u] + 1
                    if cycle_len < target_girth:
                        return False
    return True

def _construct_luw(k, g, d):
    """Construct LUW graph with parameters k, g, d. Returns (n, edges) or None if invalid."""
    q = k
    # Rule generation as in original, but we need to ensure it's correct for the given d
    # The original code had a complex rule generation.
    # Standard LUW rules for q=3:
    # y_c = x_c + x_{c-1} * x_{c-2} ? No, it's specific.
    # Let's stick to the original rule generation logic but ensure d is passed correctly.
    
    # The original _luw_construct calculated d based on g.
    # Here we take d as input.
    
    # Re-implementing the neighbor logic from the original seed, but parameterized by d.
    # The original code:
    # rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    # for c in range(5, d + 1):
    #     r, c0 = (c - 5) % 4, c - (c - 5) % 4
    #     rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    # This rule generation seems to be for a specific family.
    # We will use it.
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    rule = {}
    if d >= 2: rule[2] = (1, 1)
    if d >= 3: rule[3] = (2, 1)
    if d >= 4: rule[4] = (1, 2)
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
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
    
    n = len(order)
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Verify girth
    if _has_girth_at_least(adj, n, g):
        return n, edges
    return None

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0
    
    if k == 3 and g == 16:
        # Try LUW with increasing d until we find a valid graph
        # Minimum d for girth 16?
        # Girth is roughly 2d. So d ~ 8.
        # Let's try d from 7 to 12.
        for d in range(4, 13):
            if time.time() > deadline - 10:
                break
            result = _construct_luw(k, g, d)
            if result is not None:
                return result
        
        # Fallback to original large d if all fail (should not happen if girth logic is correct)
        # But just in case, use a large d that is guaranteed to work?
        # The original code used d=11.
        result = _construct_luw(k, g, 11)
        if result is not None:
            return result
            
    # Generic fallback
    # If k, g are not 3, 16, use the original logic or a simple LUW
    # For now, just use LUW with a reasonable d
    d = g // 2 + 2
    result = _construct_luw(k, g, d)
    if result is not None:
        return result
    
    # If all fails, return a trivial graph? No, must be valid.
    # The task implies we should always find one.
    # We'll raise an exception if we can't, but we should be safe.
    raise Exception("Failed to construct graph")
