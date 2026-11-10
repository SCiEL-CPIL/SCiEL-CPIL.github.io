# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def construct(k, g):
    deadline = time.time() + 115.0
    if k != 3:
        return _lww_fallback(k, g)
    
    best_result = _lww_fallback(3, g)
    if time.time() > deadline:
        return best_result
    
    # Try to improve via randomized search with local optimization
    # Note: LWW construction is already quite efficient for this parameter set.
    # Randomized search on such large graphs is unlikely to find a smaller one quickly
    # without significant overhead, so we skip it to ensure we stay within time limits.
    return best_result

def _lww_fallback(k, g):
    q = k
    # Choose d so that girth >= g
    # For odd d: girth >= d + 5
    # For even d: girth >= d + 4
    if (g - 5) % 2 != 0:
        d = max(3, g - 5)  # odd d, girth >= d+5 >= g
    else:
        d = max(4, g - 4)  # even d, girth >= d+4 >= g
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add_fn = lambda x, y: x ^ y
        sub_fn = lambda x, y: x ^ y
        mul_fn = lambda x, y: table[x][y]
    else:
        add_fn = lambda x, y: (x + y) % q
        sub_fn = lambda x, y: (x - y) % q
        mul_fn = lambda x, y: x * y % q
    
    # Build rule table
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)
    
    # Verify girth by checking non-backtracking walks
    def _verify_girth(n, edges, g):
        """Check that no cycle shorter than g exists using BFS from each vertex."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Check degree
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        # Check for short cycles using BFS with parent tracking
        for start in range(n):
            # BFS to find shortest cycle through start
            visited = {start: (-1, 0)}  # node: (parent, dist)
            queue = deque([(start, -1, 0)])
            found_cycle = False
            while queue and not found_cycle:
                node, parent, dist = queue.popleft()
                if dist >= g // 2:
                    continue  # No need to go deeper if we're already at half the girth
                for nb in adj[node]:
                    if nb == parent:
                        continue
                    if nb in visited:
                        # Found a cycle
                        cycle_len = dist + 1 + visited[nb][1]
                        if cycle_len < g:
                            return False
                    else:
                        visited[nb] = (node, dist + 1)
                        queue.append((nb, node, dist + 1))
        return True
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add_fn(x[c], mul_fn(y[a], x[b]))
                else:
                    y[c] = sub_fn(x[c], mul_fn(x[a], y[b]))
            yield 1 - side, tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    n = len(order)
    
    # Fast verification: check degrees and simple structure.
    # We trust the LWW construction for girth guarantees to avoid O(N^2) verification.
    adj = [[] for _ in range(n)]
    for u, v in edges:
        if u == v:
            return None # Loop
        adj[u].append(v)
        adj[v].append(u)
        
    for i in range(n):
        if len(adj[i]) != k:
            return None

    # Check for duplicate edges
    edge_set = set()
    for u, v in edges:
        if u > v: u, v = v, u
        if (u, v) in edge_set:
            return None
        edge_set.add((u, v))

    return n, edges

def _try_lww_variant(k, g, d):
    """Try a different LWW parameterization."""
    q = k
    if (g - 5) % 2 != 0:
        d = max(3, g - 5)
    else:
        d = max(4, g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add_fn = lambda x, y: x ^ y
        sub_fn = lambda x, y: x ^ y
        mul_fn = lambda x, y: table[x][y]
    else:
        add_fn = lambda x, y: (x + y) % q
        sub_fn = lambda x, y: (x - y) % q
        mul_fn = lambda x, y: x * y % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add_fn(x[c], mul_fn(y[a], x[b]))
                else:
                    y[c] = sub_fn(x[c], mul_fn(x[a], y[b]))
            yield 1 - side, tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    n = len(order)
    
    # Quick verification
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for i in range(n):
        if len(adj[i]) != k:
            return None
    
    return n, edges

def _randomized_cage(k, g, deadline, best_result):
    """
    Placeholder for randomized search.
    Given the size of the LWW graph (~350k vertices), randomized local search
    is computationally expensive and unlikely to beat the known 936-vertex cage
    within the time limit without a specialized algorithm (like the one by Exoo et al.).
    We return None to indicate no improvement found quickly.
    """
    return None

# EVOLVE-BLOCK-END
