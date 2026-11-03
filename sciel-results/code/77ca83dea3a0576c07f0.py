import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin

    if k != 3:
        # Fallback for other k, though task is k=3
        return seed_fallback(k, g)

    # Try to construct a graph using LUW with q=2 (smaller size)
    # Moore bound for k=3, g=16 is 510.
    # LUW with q=2, d=8 gives n=512.
    # LUW with q=2, d=9 gives n=1024.
    
    best_n = float('inf')
    best_edges = None

    # Try q=2 with d from 8 upwards
    # The girth of LUW(q, d) is typically 2d or slightly less/more depending on collisions.
    # We need girth >= 16.
    for d in range(8, 12):
        if time.time() > deadline - 10:
            break
        n, edges = luw_construct(2, d)
        if not edges:
            continue
        # Check girth
        if check_girth(edges, n, g):
            if n < best_n:
                best_n = n
                best_edges = edges
            # Since n grows with d, if we find one, we can stop if we assume monotonicity?
            # Not necessarily, but usually larger d -> larger n.
            # So the first valid one (smallest d) is likely the best.
            break

    if best_edges:
        return best_n, best_edges

    # Fallback to q=3 if q=2 failed or took too long
    for d in range(8, 12):
        if time.time() > deadline - 5:
            break
        n, edges = luw_construct(3, d)
        if not edges:
            continue
        if check_girth(edges, n, g):
            if n < best_n:
                best_n = n
                best_edges = edges
            break

    if best_edges:
        return best_n, best_edges

    # Final fallback: Original seed logic (q=3, d=11)
    return seed_fallback(3, g)

def luw_construct(q, d):
    """
    Construct a LUW graph with parameters q and d.
    Returns (n, edges).
    """
    # LUW construction details:
    # Vertices are (side, x) where side in {0, 1}, x is a tuple of length d over F_q.
    # Actually, the standard LUW graph G(q, d) has 2 * q^d vertices?
    # Let's implement the standard recurrence.
    
    # Rule for LUW:
    # y[1] = x[1]
    # y[c] = x[c] + y[c-1] * x[c-1] ? No, that's for specific cases.
    
    # The seed used:
    # rule[c] = (a, b)
    # y[c] = x[c] + y[a] * x[b] (side 0)
    # y[c] = x[c] - x[a] * y[b] (side 1)
    
    # For q=2, subtraction is addition.
    
    # We need to define the rule.
    # The seed defined a rule based on c.
    # Let's reuse the seed's rule generation logic but parameterized by q.
    
    # Seed rule logic:
    # rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    # for c in 5..d:
    #   r, c0 = (c - 5) % 4, c - (c - 5) % 4
    #   rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    # This rule seems specific to the seed's choice.
    # We will use this same rule structure.
    
    def get_rule(c, d):
        if c == 2: return (1, 1)
        if c == 3: return (2, 1)
        if c == 4: return (1, 2)
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        return [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    add = lambda x, y: (x + y) % q
    sub = lambda x, y: (x - y) % q
    mul = lambda x, y: (x * y) % q

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [0] * (d + 1) # 1-indexed, y[0] unused
            y[1] = first
            for c in range(2, d + 1):
                a, b = get_rule(c, d)
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y[1:])

    start = (0, (0,) * d)
    index = {}
    order = []
    edges = []
    
    index[start] = 0
    order.append(start)
    
    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges

def check_girth(edges, n, g):
    """
    Check if the graph has girth >= g.
    Uses BFS from each node.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Check for self loops
    for u, v in edges:
        if u == v:
            return False
            
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            if dist[u] >= g // 2:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def seed_fallback(k, g):
    # Original fallback logic, kept for compatibility
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # ... (same as before but using luw_construct?)
    # The original seed_fallback was a bit different in implementation details (indexing).
    # We will just call luw_construct with the original parameters.
    return luw_construct(q, d)
