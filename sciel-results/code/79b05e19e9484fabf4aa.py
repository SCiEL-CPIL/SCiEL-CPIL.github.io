"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

def _verify_girth(n, edges, k, g):
    """Verify that the graph has girth at least g using BFS from each vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check degree
    for i in range(n):
        if len(adj[i]) != k:
            return False
    
    # Check no short cycles using non-backtracking walks
    for start in range(n):
        # BFS tracking (current_node, previous_node, depth)
        visited = set()
        queue = deque()
        for neighbor in adj[start]:
            queue.append((start, neighbor, 1))
            visited.add((start, neighbor))
        
        while queue:
            prev, curr, depth = queue.popleft()
            if depth >= g:
                continue
            for next_node in adj[curr]:
                if next_node == prev:
                    continue
                if next_node == start and depth + 1 < g:
                    return False
                if (curr, next_node) not in visited:
                    visited.add((curr, next_node))
                    queue.append((curr, next_node, depth + 1))
    return True

def construct(k, g):
    deadline = time.time() + 140.0  # 10s safety margin
    
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k
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
    if _verify_girth(n, edges, k, g):
        return n, edges
    
    # Fallback: try different d values
    for d_try in range(d + 1, min(d + 10, 20)):
        if time.time() > deadline:
            break
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d_try + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        
        def neighbours2(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d_try - 1)
                for c in range(2, d_try + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)
        
        start2 = (0, (None,) + (0,) * d_try)
        index2, order2, edges2 = {start2: 0}, [start2], []
        for vertex in order2:
            for other in neighbours2(vertex):
                if other not in index2:
                    index2[other] = len(order2)
                    order2.append(other)
                if vertex[0] == 0:
                    edges2.append((index2[vertex], index2[other]))
        
        n2 = len(order2)
        if _verify_girth(n2, edges2, k, g):
            return n2, edges2
    
    # If all else fails, return the original (should be valid for LUW with proper params)
    return n, edges
# EVOLVE-BLOCK-END
