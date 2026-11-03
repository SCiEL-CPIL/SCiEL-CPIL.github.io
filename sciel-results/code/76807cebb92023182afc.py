"""Rewritten program using optimized LUW construction parameters."""
# EVOLVE-BLOCK-START
import time
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g on as few vertices as possible.
    Optimizes the Lazebnik-Ustimenko-Woldar (LUW) construction parameters.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Calculate optimal d for LUW construction
    # Girth bound for LUW(d, q) is 2d + 2
    # We need 2d + 2 >= g => d >= (g - 2) / 2
    d_min = (g - 2 + 1) // 2  # ceil((g-2)/2)
    if d_min < 3:
        d_min = 3
        
    # For k=4, q=4 is a prime power.
    # For general k, we check if k is a prime power.
    # If not, we might need to fall back or use a different construction.
    # However, the task context is specifically k=4, g=10.
    # We will handle k=4 explicitly with q=4.
    
    if k == 4:
        q = 4
        # Use the minimal d that satisfies the girth requirement
        d = d_min
        
        result = _luw_construct(q, d)
        if result is not None:
            n, edges = result
            # Verify the girth to be absolutely sure
            # The verification is fast for n=512
            if _has_girth_at_least(edges, n, g):
                return n, edges
            else:
                # If verification fails (shouldn't happen for correct LUW), try larger d
                for d_try in range(d+1, 10):
                    result_try = _luw_construct(q, d_try)
                    if result_try is not None:
                        n_try, edges_try = result_try
                        if _has_girth_at_least(edges_try, n_try, g):
                            return n_try, edges_try
                # Fallback to original seed logic if all else fails
                return _luw_construct_original(k, g)
        
    # Fallback for other k values
    # Try to find a prime power q such that q == k or use generic logic
    # For now, just use the original logic which assumes q=k
    return _luw_construct_original(k, g)

def _luw_construct(q, d):
    """
    Lazebnik-Ustimenko-Woldar graph construction with specific q and d.
    Returns (n, edges) or None if invalid.
    """
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        # Assume q is prime for simplicity in fallback, though LUW works for prime powers.
        # For this task k=4, so q=4 is the main path.
        if q < 2:
            return None
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    # Define the rule for LUW construction
    rule = {}
    if d >= 2:
        rule[2] = (1, 1)
    if d >= 3:
        rule[3] = (2, 1)
    if d >= 4:
        rule[4] = (1, 2)
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        
    # Ensure all required c in 2..d are in rule
    for c in range(2, d+1):
        if c not in rule:
            rule[c] = (1, c-1)

    def neighbours(vertex):
        side, x = vertex
        # x is a tuple of length d+1: (None, x_1, x_2, ..., x_d)
        for first in range(q):
            y_list = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y_list[c] = add(x[c], mul(y_list[a], x[b]))
                else:
                    y_list[c] = sub(x[c], mul(x[a], y_list[b]))
            yield 1 - side, tuple(y_list)
    
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

def _luw_construct_original(k, g):
    """Original seed logic for fallback."""
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

def _has_girth_at_least(edges, n, g):
    """
    Check if graph has girth at least g using BFS from each vertex.
    Returns True if no cycle of length < g is found.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            if dist[u] >= (g - 1) // 2:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    if v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
    return True

# EVOLVE-BLOCK-END
